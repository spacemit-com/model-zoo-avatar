#include "feathertalk_service.h"
#include "internal.h"
#include <yaml-cpp/yaml.h>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <map>
#include <sstream>
#include <memory>
#include <algorithm>
#include <cstdlib>
#include <csignal>
#include <charconv>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#ifdef __linux__
#include <sched.h>
#endif
namespace fs=std::filesystem;
using namespace feathertalk;
namespace {
volatile std::sig_atomic_t stopped=0;
void on_signal(int){stopped=1;}
void check_cancel(){if(stopped)throw std::runtime_error("Worker cancelled; inference returned before releasing resources");}
class OutputDirectory {
public:
    explicit OutputDirectory(const fs::path& path) {
        fs::create_directories(path);
        fd_ = open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (fd_ < 0) throw std::runtime_error("Cannot open output directory");
        try {
            if (flock(fd_, LOCK_EX | LOCK_NB) != 0) throw std::runtime_error("Output directory is in use");
            if (!fs::is_empty(path)) throw std::runtime_error("Output directory must be empty");
        } catch (...) {
            close(fd_);
            throw;
        }
    }
    ~OutputDirectory() { close(fd_); }
    OutputDirectory(const OutputDirectory&) = delete;
    OutputDirectory& operator=(const OutputDirectory&) = delete;
private:
    int fd_;
};
}
std::string path(const std::string& value,const fs::path& folder){
    if(value.empty())throw std::invalid_argument("Empty resource path");
    if(value.rfind("~/",0)==0){const char* home=std::getenv("HOME");if(!home)throw std::runtime_error("HOME unset");return (fs::path(home)/value.substr(2)).string();}
    fs::path p(value);return (p.is_absolute()?p:folder/p).lexically_normal().string();
}
RuntimeOptions runtime(YAML::Node n,const fs::path& parent){
    RuntimeOptions o;
    if(n["ort_library"])o.ort_library=n["ort_library"].as<std::string>();
    if(n["ep_library"])o.ep_library=n["ep_library"].as<std::string>();
    for(auto* p:{&o.ort_library,&o.ep_library})if(p->find('/')!=std::string::npos)*p=path(*p,parent);
    if(n["provider"])o.provider=n["provider"].as<std::string>();
    if(n["threads"])o.threads=n["threads"].as<int>();
    if(n["ep_threads"])o.ep_threads=n["ep_threads"].as<int>();
    if(n["ep_affinity"])o.ep_affinity=n["ep_affinity"].as<std::string>();
    if(n["profile_prefix"])o.profile_prefix=path(n["profile_prefix"].as<std::string>(),parent);
    return o;
}
void affinity(const YAML::Node& node){
#ifdef __linux__
    if(!node["cpu_affinity"])return;
    cpu_set_t mask;CPU_ZERO(&mask);const auto ids=node["cpu_affinity"].as<std::vector<int>>();
    if(ids.empty())throw std::invalid_argument("Empty CPU affinity");
    for(int id:ids){if(id<0||id>=CPU_SETSIZE)throw std::invalid_argument("CPU id out of range");CPU_SET(id,&mask);}
    if(sched_setaffinity(0,sizeof(mask),&mask))throw std::runtime_error("Cannot apply CPU affinity");
#else
    if(node["cpu_affinity"])throw std::runtime_error("Affinity requires Linux");
#endif
}
int main(int argc,char** argv){
 std::signal(SIGTERM,on_signal);std::signal(SIGINT,on_signal);
 try{
    if(argc<2)throw std::invalid_argument("Usage: feathertalk_worker extract|render --config FILE --audio WAV --features FTF --output PATH [--max-frames N]");
    const std::string mode=argv[1];std::map<std::string,std::string> args;
    for(int i=2;i<argc;i+=2){if(i+1>=argc)throw std::invalid_argument("Missing option value");const std::string key=argv[i];if(key!="--config"&&key!="--audio"&&key!="--features"&&key!="--output"&&key!="--max-frames")throw std::invalid_argument("Unknown option");args[key]=argv[i+1];}
    if(!args.count("--config")||!args.count("--output"))throw std::invalid_argument("config and output required");
    const auto file=fs::absolute(args.at("--config"));const auto folder=file.parent_path();auto config=YAML::LoadFile(file.string());
    if(!config["schema"]||config["schema"].as<int>()!=1)throw std::runtime_error("Expected config schema=1");
    const auto start=std::chrono::steady_clock::now();std::uint64_t count=0;
    if(mode=="extract"){
        if(!args.count("--audio"))throw std::invalid_argument("audio required");auto n=config["hubert"];affinity(n);auto o=runtime(n,folder);
        FeatherHubertExtractor model(path(n["model"].as<std::string>(),folder),o);
        auto pcm=ReadWav16k(args.at("--audio"));detail::FeatureWriter writer(args.at("--output"));
        for(std::size_t offset=0;offset<pcm.size();offset+=12000){check_cancel();const auto size=std::min<std::size_t>(12000,pcm.size()-offset);auto batch=model.PushPcm(pcm.data()+offset,size,offset+size==pcm.size());check_cancel();count+=batch.size();writer.Append(batch);}
        if(!count)throw std::runtime_error("Audio is too short to produce feature windows");
        // The extractor preserves the original server's model-frame count.
        // Pad the final video ticks with closed-mouth features to cover the PCM duration.
        const auto needed=(pcm.size()+639)/640;
        if(count>needed)throw std::runtime_error("Feature count exceeds audio timeline");
        if(count<needed){FeatureBatch tail;for(auto i=count;i<needed;++i)tail.frame_ids.push_back(i);tail.values.resize(tail.size()*kWindowElements,0);writer.Append(tail);count=needed;}
        writer.Finish(pcm.size());
    }else if(mode=="render"){
        if(!args.count("--features"))throw std::invalid_argument("features required");auto n=config["unet"];affinity(n);auto o=runtime(n,folder);
        UnetLipGenerator model(path(n["model"].as<std::string>(),folder),o);
        const auto dataset=fs::path(path(config["assets"]["person"].as<std::string>(),folder));
        std::vector<std::pair<fs::path,fs::path>> frames;
        for(const auto& e:fs::directory_iterator(dataset/"full_body_img")){
            if(e.path().extension()!=".jpg")continue;const auto stem=e.path().stem().string();
            if(stem.empty()||!std::all_of(stem.begin(),stem.end(),[](unsigned char c){return c>='0'&&c<='9';}))continue;
            const auto landmarks=dataset/"landmarks"/(stem+".lms");if(!fs::is_regular_file(landmarks))throw std::runtime_error("Frame missing landmarks");frames.emplace_back(e.path(),landmarks);
        }
        std::sort(frames.begin(),frames.end(),[](const auto& a,const auto& b){return std::stoull(a.first.stem().string())<std::stoull(b.first.stem().string());});
        if(frames.empty())throw std::runtime_error("No paired JPG/landmark frames");
        RenderOptions ro;auto r=config["render"];
        if(r){
            if(r["native_pasteback"])ro.native_pasteback=r["native_pasteback"].as<bool>();
            if(r["mouth_only_blend"])ro.mouth_only_blend=r["mouth_only_blend"].as<bool>();
            if(r["feather"])ro.feather=r["feather"].as<bool>();
            if(r["mouth_sharpen"])ro.mouth_sharpen=r["mouth_sharpen"].as<float>();
            if(r["temporal_smooth"])ro.temporal_smooth=r["temporal_smooth"].as<bool>();
            if(r["smooth_alpha"])ro.smooth_alpha=r["smooth_alpha"].as<float>();
            if(r["patch_lowpass"])ro.patch_lowpass=r["patch_lowpass"].as<float>();
        }
        std::uint64_t limit=0;
        if (args.count("--max-frames")) {
            const auto& value=args.at("--max-frames");
            const auto parsed=std::from_chars(value.data(),value.data()+value.size(),limit);
            if(parsed.ec!=std::errc{} || parsed.ptr!=value.data()+value.size())throw std::invalid_argument("max-frames must be nonnegative");
        }
        const OutputDirectory output_directory(args.at("--output"));
        detail::FeatureReader reader(args.at("--features"));std::vector<float> window;std::unique_ptr<PreparedFrame> frame;
        while((!limit||count<limit)&&reader.Next(window)){
            check_cancel();
            const auto period=frames.size()==1?1:2*frames.size()-2;const auto at=count%period;const auto index=at<frames.size()?at:period-at;
            auto image=cv::imread(frames[index].first.string(),cv::IMREAD_COLOR);auto landmarks=ReadLandmarks(frames[index].second.string());
            if(!frame)frame=std::make_unique<PreparedFrame>(image,landmarks,ro);else frame->SetFrame(image,landmarks,ro);
            auto output=model.Infer(frame->ImageTensor(),window);check_cancel();auto composed=frame->Compose(output);
            std::ostringstream name;name<<std::setw(8)<<std::setfill('0')<<count<<".png";
            if(!cv::imwrite((fs::path(args.at("--output"))/name.str()).string(),composed))throw std::runtime_error("Image write failed");++count;
        }
    }else throw std::invalid_argument("Unknown mode");
    const auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    std::cout<<"{\"ok\":true,\"frames\":"<<count<<",\"elapsed_ms\":"<<ms<<"}\n";return 0;
 }catch(const std::exception& e){std::cerr<<"feathertalk: "<<e.what()<<'\n';return 2;}
}
