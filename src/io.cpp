#include "internal.h"
#include <array>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
namespace {
std::uint64_t le(const unsigned char* p,int n){std::uint64_t x=0;for(int i=0;i<n;++i)x|=std::uint64_t(p[i])<<(8*i);return x;}
void put(unsigned char* p,std::uint64_t x,int n){for(int i=0;i<n;++i)p[i]=static_cast<unsigned char>(x>>(8*i));}
void exact(std::istream& s,char* p,std::size_t n){if(!s.read(p,static_cast<std::streamsize>(n)))throw std::runtime_error("Truncated input file");}
void little_host(){std::uint16_t x=1;if(*reinterpret_cast<unsigned char*>(&x)!=1)throw std::runtime_error("Little-endian host required");}
}
namespace SpacemiT::avatar {
std::vector<std::int16_t> ReadWav16k(const std::string& path){
    std::ifstream s(path,std::ios::binary);if(!s)throw std::runtime_error("Cannot open WAV: "+path);
    std::array<unsigned char,12> head{};exact(s,reinterpret_cast<char*>(head.data()),12);
    if(std::memcmp(head.data(),"RIFF",4)||std::memcmp(head.data()+8,"WAVE",4))throw std::runtime_error("Expected RIFF WAVE");
    const auto file_size=std::filesystem::file_size(path);
    const auto riff_end = le(head.data()+4,4)+8;
    if(riff_end<12 || riff_end>file_size)throw std::runtime_error("Invalid RIFF length");
    bool fmt=false;std::vector<std::int16_t> pcm;
    while(static_cast<std::uint64_t>(s.tellg())<riff_end){
        if (riff_end-static_cast<std::uint64_t>(s.tellg())<8) throw std::runtime_error("Truncated WAV chunk header");
        std::array<unsigned char,8> h{};exact(s,reinterpret_cast<char*>(h.data()),8);const auto n=le(h.data()+4,4);
        const auto at=static_cast<std::uint64_t>(s.tellg());if(n+(n&1)>riff_end-at)throw std::runtime_error("Invalid WAV chunk length");
        if(!std::memcmp(h.data(),"fmt ",4)){
            if(n<16)throw std::runtime_error("Short WAV fmt");
            std::array<unsigned char,16> f{};exact(s,reinterpret_cast<char*>(f.data()),16);
            if(le(f.data(),2)!=1||le(f.data()+2,2)!=1||le(f.data()+4,4)!=16000||le(f.data()+14,2)!=16||le(f.data()+12,2)!=2||le(f.data()+8,4)!=32000)throw std::runtime_error("WAV must be 16kHz mono PCM16");
            fmt=true;s.seekg(static_cast<std::streamoff>(n-16),std::ios::cur);
        }else if(!std::memcmp(h.data(),"data",4)){
            if(!fmt || n%2 || n>512*1024*1024 || pcm.size()*2+n>512*1024*1024)throw std::runtime_error("Invalid or excessive WAV data");
            const auto before=pcm.size();pcm.resize(before+n/2);little_host();exact(s,reinterpret_cast<char*>(pcm.data()+before),n);
        }else s.seekg(static_cast<std::streamoff>(n),std::ios::cur);
        if(n&1)s.seekg(1,std::ios::cur);
    }
    if(!fmt||pcm.empty())throw std::runtime_error("WAV contains no PCM audio");
    return pcm;
}
}
namespace SpacemiT::avatar::detail {
FeatureWriter::FeatureWriter(const std::string& path){
    little_host();
    stream_.reset(std::fopen(path.c_str(), "wbx"));
    if(!stream_)throw std::runtime_error("Cannot exclusively create feature output: "+path);
    std::array<char,64> h{};
    if (std::fwrite(h.data(), 1, h.size(), stream_.get()) != h.size())
        throw std::runtime_error("Feature header write failed");
}
void FeatureWriter::Append(const FeatureBatch& b){
    if(!stream_)throw std::runtime_error("Feature output already finalized");
    if(b.values.size()!=b.size()*kWindowElements)throw std::runtime_error("Invalid feature batch");
    for(std::size_t i=0;i<b.size();++i)
        if(b.frame_ids[i]!=static_cast<std::int64_t>(count_+i))throw std::runtime_error("Feature stream has missing or duplicate frame ids");
    if(!b.values.empty() && std::fwrite(b.values.data(),sizeof(float),b.values.size(),stream_.get())!=b.values.size())
        throw std::runtime_error("Failed to write features");
    count_+=b.size();
}
void FeatureWriter::Finish(std::uint64_t samples){
    if(!stream_ || count_!=samples/640+(samples%640!=0))throw std::runtime_error("Invalid feature timeline");
    std::array<unsigned char,64> h{};std::memcpy(h.data(),"FTF1",4);put(h.data()+4,1,4);put(h.data()+8,16000,4);put(h.data()+12,25,4);put(h.data()+16,count_,8);put(h.data()+24,samples,8);
    if(std::fseek(stream_.get(),0,SEEK_SET)!=0 || std::fwrite(h.data(),1,h.size(),stream_.get())!=h.size() || std::fflush(stream_.get())!=0)
        throw std::runtime_error("Feature header write failed");
    stream_.reset();
}
FeatureReader::FeatureReader(const std::string& path):stream_(path,std::ios::binary){
    little_host();if(!stream_)throw std::runtime_error("Cannot open feature file");std::array<unsigned char,64> h{};exact(stream_,reinterpret_cast<char*>(h.data()),64);
    if(std::memcmp(h.data(),"FTF1",4)||le(h.data()+4,4)!=1||le(h.data()+8,4)!=16000||le(h.data()+12,4)!=25)throw std::runtime_error("Invalid FTF contract");
    count_=le(h.data()+16,8);if(count_>(std::numeric_limits<std::uint64_t>::max()-64)/(kWindowElements*4)||std::filesystem::file_size(path)!=64+count_*kWindowElements*4)throw std::runtime_error("FTF file size mismatch");
    const auto samples=le(h.data()+24,8);
    if(count_!=samples/640+(samples%640!=0))throw std::runtime_error("Invalid FTF timeline");
}
bool FeatureReader::Next(std::vector<float>& w){if(read_==count_)return false;w.resize(kWindowElements);exact(stream_,reinterpret_cast<char*>(w.data()),w.size()*4);++read_;return true;}
}
