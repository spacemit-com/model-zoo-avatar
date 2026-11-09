#include "runtime.h"
#include "internal.h"
#include "non_reentrant.h"
#include <cmath>
#include <stdexcept>
#include <array>
namespace feathertalk {
struct FeatherHubertExtractor::Impl {
    std::mutex mutex;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<detail::FeatureStream> stream;
    Impl(const std::string& model,const RuntimeOptions& o) {
        if (o.provider!="cpu") throw std::invalid_argument("FeatherHuBERT baseline requires CPU provider");
        detail::InitRuntime(o.ort_library); auto so=detail::SessionOptions(o);
        session=std::make_unique<Ort::Session>(detail::Environment(),model.c_str(),so);
        if (session->GetInputCount()!=1 || session->GetOutputCount()!=1) throw std::runtime_error("Invalid HuBERT graph signature");
        stream=std::make_unique<detail::FeatureStream>([this](const std::vector<float>& audio) {
            std::array<std::int64_t,2> shape{1,static_cast<std::int64_t>(audio.size())};
            auto mem=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
            auto input=Ort::Value::CreateTensor<float>(mem,const_cast<float*>(audio.data()),audio.size(),shape.data(),shape.size());
            const char* in[]={"waveform"}; const char* out[]={"hidden"};
            auto values=session->Run(Ort::RunOptions{nullptr},in,&input,1,out,1);
            auto info=values[0].GetTensorTypeAndShapeInfo(); auto dims=info.GetShape();
            if (info.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || dims.size()!=3 || dims[0]!=1 || dims[2]!=1024) throw std::runtime_error("Invalid HuBERT hidden shape");
            auto count=info.GetElementCount();const float* data=values[0].GetTensorData<float>();
            for (std::size_t i=0;i<count;++i) if (!std::isfinite(data[i])) throw std::runtime_error("Nonfinite HuBERT features");
            return std::vector<float>(data,data+count);
        });
    }
};
FeatherHubertExtractor::FeatherHubertExtractor(const std::string& model,const RuntimeOptions& o):impl_(std::make_unique<Impl>(model,o)){}
FeatherHubertExtractor::~FeatherHubertExtractor()=default;
FeatureBatch FeatherHubertExtractor::PushPcm(const std::int16_t* p,std::size_t n,bool f){auto gate=detail::Enter(impl_->mutex);return impl_->stream->Push(p,n,f);}
void FeatherHubertExtractor::Reset(){auto gate=detail::Enter(impl_->mutex);impl_->stream->Reset();}
struct UnetLipGenerator::Impl {
    std::mutex mutex;
    std::unique_ptr<Ort::Session> session;
    Impl(const std::string& model,const RuntimeOptions& o) {
        detail::InitRuntime(o.ort_library); auto so=detail::SessionOptions(o);
        session=std::make_unique<Ort::Session>(detail::Environment(),model.c_str(),so);
        if(session->GetInputCount()!=2 || session->GetOutputCount()!=1)throw std::runtime_error("Invalid UNet graph signature");
    }
};
UnetLipGenerator::UnetLipGenerator(const std::string& model,const RuntimeOptions& o):impl_(std::make_unique<Impl>(model,o)){}
UnetLipGenerator::~UnetLipGenerator()=default;
std::vector<float> UnetLipGenerator::Infer(const std::vector<float>& image,const std::vector<float>& audio){
    auto gate=detail::Enter(impl_->mutex);
    if(image.size()!=6*160*160 || audio.size()!=kWindowElements)throw std::invalid_argument("UNet expects image [1,6,160,160], audio [1,16,32,32]");
    for(float x:image)if(!std::isfinite(x))throw std::invalid_argument("Nonfinite image tensor");
    for(float x:audio)if(!std::isfinite(x))throw std::invalid_argument("Nonfinite audio tensor");
    auto mem=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    const std::array<std::int64_t,4> a{1,6,160,160},b{1,16,32,32};
    std::array<Ort::Value,2> inputs{
        Ort::Value::CreateTensor<float>(mem,const_cast<float*>(image.data()),image.size(),a.data(),4),
        Ort::Value::CreateTensor<float>(mem,const_cast<float*>(audio.data()),audio.size(),b.data(),4)};
    const char* in[]={"input","audio"};const char* out[]={"output"};
    auto result=impl_->session->Run(Ort::RunOptions{nullptr},in,inputs.data(),2,out,1);
    detail::CheckTensor(result[0],{1,3,160,160});const auto* data=result[0].GetTensorData<float>();
    for(std::size_t i=0;i<3*160*160;++i)if(!std::isfinite(data[i]))throw std::runtime_error("Nonfinite UNet output");
    return {data,data+3*160*160};
}
}
