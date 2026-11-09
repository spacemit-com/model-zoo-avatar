#include "feathertalk_service.h"
#include "internal.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <chrono>
using namespace feathertalk;
void require(bool x,const char* message){if(!x)throw std::runtime_error(message);}
int main(){
 try{
    auto runner=[](const std::vector<float>& a){return std::vector<float>((a.size()/320)*1024,0.5f);};
    detail::FeatureStream stream(runner);std::vector<std::int16_t> silence(24000,0);
    auto first=stream.Push(silence.data(),12000,false);auto last=stream.Push(silence.data()+12000,12000,true);
    require(first.size()>0&&last.size()>0,"No silent frames");
    require(std::all_of(first.values.begin(),first.values.end(),[](float v){return v==0;}),"Silence amplified");
    require(first.frame_ids.front()==0&&last.frame_ids.front()==first.frame_ids.back()+1,"Frame ids not continuous");
    bool finalized=false;try{stream.Push(nullptr,0,false);}catch(const std::runtime_error&){finalized=true;}require(finalized,"Push after final accepted");
    stream.Reset();std::vector<std::int16_t> speech(12000);for(std::size_t i=0;i<speech.size();++i)speech[i]=static_cast<std::int16_t>(12000*std::sin(i*0.1));
    auto voice=stream.Push(speech.data(),speech.size(),true);require(voice.frame_ids.front()==0,"Reset cursor failed");require(std::any_of(voice.values.begin(),voice.values.end(),[](float v){return v!=0;}),"Speech discarded");
    int calls=0;
    detail::FeatureStream retry([&](const std::vector<float>& samples) {
        if (++calls==2) throw std::runtime_error("Injected inference failure");
        return runner(samples);
    });
    bool failed=false;
    try { retry.Push(silence.data(),silence.size(),true); }
    catch (const std::runtime_error&) { failed=true; }
    require(failed,"Injected failure not observed");
    const auto retried=retry.Push(silence.data(),silence.size(),true);
    require(retried.size()==first.size()+last.size() && retried.frame_ids.front()==0,
            "Failed multi-chunk push consumed audio or advanced the cursor");
    cv::Mat image(240,240,CV_8UC3);for(int y=0;y<240;++y)for(int x=0;x<240;++x)image.at<cv::Vec3b>(y,x)={static_cast<unsigned char>(x),static_cast<unsigned char>(y),static_cast<unsigned char>((x+y)/2)};
    std::vector<cv::Point2f> lm(110,{120,120});lm[1]={30,120};lm[31]={210,120};lm[52]={120,30};for(int i=90;i<110;++i)lm[i]={100.0f+(i-90)*2.0f,150.0f+static_cast<float>(i%3)};
    PreparedFrame frame(image,lm);auto t=frame.ImageTensor();require(t.size()==6*160*160,"Image tensor shape");
    const auto plane=160*160;require(t[3*plane+5*160+5]==0,"Mask start wrong");require(t[3*plane+149*160+154]==0,"Mask end wrong");require(t[3*plane+150*160+155]>0,"Mask extended past boundary");
    auto output=std::vector<float>(t.begin(),t.begin()+3*plane);auto composed=frame.Compose(output);
    require(composed.type()==CV_8UC3&&composed.size()==image.size(),"Composite shape");require(composed.at<cv::Vec3b>(0,0)==image.at<cv::Vec3b>(0,0),"Modified outside face");
    bool invalid=false;try{frame.Compose({0});}catch(const std::invalid_argument&){invalid=true;}require(invalid,"Bad output shape accepted");
    const auto temp=std::filesystem::temp_directory_path()/("ft-core-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    require(std::filesystem::create_directory(temp),"Test directory exists");const auto file=temp/"features.ftf";
    {detail::FeatureWriter writer(file.string());FeatureBatch b;b.frame_ids={0};b.values.resize(kWindowElements,0.25f);writer.Append(b);writer.Finish(640);}
    bool preserved=false;try{detail::FeatureWriter writer(file.string());}catch(const std::runtime_error&){preserved=true;}require(preserved,"Existing feature file overwritten");
    {detail::FeatureReader reader(file.string());std::vector<float> w;require(reader.Count()==1&&reader.Next(w)&&w.size()==kWindowElements&&w[0]==0.25f&&!reader.Next(w),"Feature file roundtrip failed");}
    std::filesystem::remove(file);std::filesystem::remove(temp);
    std::cout<<"feature cursor, final/reset, silence guard, mask bounds, composite and invalid input: passed\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
