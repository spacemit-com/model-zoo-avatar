#include "feathertalk_service.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
namespace py=pybind11;
using namespace feathertalk;
template<class T>void check(const py::array& a){if(!a.dtype().is(py::dtype::of<T>())||!(a.flags()&py::array::c_style))throw py::value_error("Expected exact native dtype and C-contiguous array");}
template<class T>
std::vector<T> copy(const py::array& a) {
    check<T>(a);
    // NumPy C-contiguous arrays can have an unaligned data pointer.
    std::vector<T> values(a.size());
    if (!values.empty()) std::memcpy(values.data(), a.data(), values.size() * sizeof(T));
    return values;
}
py::array_t<float> tensor(const std::vector<float>& values,std::vector<py::ssize_t> shape){py::array_t<float> a(shape);if(!values.empty())std::memcpy(a.mutable_data(),values.data(),values.size()*sizeof(float));return a;}
cv::Mat image(const py::array& a){check<std::uint8_t>(a);if(a.ndim()!=3||a.shape(2)!=3||a.shape(0)>16384||a.shape(1)>16384||a.shape(0)<1||a.shape(1)<1)throw py::value_error("Expected BGR uint8 HxWx3");return cv::Mat(static_cast<int>(a.shape(0)),static_cast<int>(a.shape(1)),CV_8UC3,const_cast<void*>(a.data())).clone();}
std::vector<cv::Point2f> points(const py::array& a) {
    if (a.ndim()!=2 || a.shape(1)!=2 || a.shape(0)<110 || a.shape(0)>10000)
        throw py::value_error("Expected float32 [N>=110,2] landmarks");
    const auto values = copy<float>(a);
    std::vector<cv::Point2f> out;
    out.reserve(a.shape(0));
    for (std::size_t i=0; i<values.size(); i+=2) out.emplace_back(values[i], values[i+1]);
    return out;
}
bool shape(const py::array& a,const std::vector<py::ssize_t>& dims){if(a.ndim()!=static_cast<py::ssize_t>(dims.size()))return false;for(std::size_t i=0;i<dims.size();++i)if(a.shape(i)!=dims[i])return false;return true;}
PYBIND11_MODULE(_feathertalk,m){
 m.doc()="FeatherTalk C++17 two-model inference; one ORT binary per process";
 py::class_<RuntimeOptions>(m,"RuntimeOptions").def(py::init<>()).def_readwrite("ort_library",&RuntimeOptions::ort_library).def_readwrite("provider",&RuntimeOptions::provider).def_readwrite("ep_library",&RuntimeOptions::ep_library).def_readwrite("ep_affinity",&RuntimeOptions::ep_affinity).def_readwrite("profile_prefix",&RuntimeOptions::profile_prefix).def_readwrite("threads",&RuntimeOptions::threads).def_readwrite("ep_threads",&RuntimeOptions::ep_threads);
 py::class_<RenderOptions>(m,"RenderOptions").def(py::init<>()).def_readwrite("feather",&RenderOptions::feather).def_readwrite("mouth_only_blend",&RenderOptions::mouth_only_blend).def_readwrite("native_pasteback",&RenderOptions::native_pasteback).def_readwrite("temporal_smooth",&RenderOptions::temporal_smooth).def_readwrite("mouth_sharpen",&RenderOptions::mouth_sharpen).def_readwrite("smooth_alpha",&RenderOptions::smooth_alpha).def_readwrite("patch_lowpass",&RenderOptions::patch_lowpass);
 py::class_<FeatureBatch>(m,"FeatureBatch").def_property_readonly("windows",[](const FeatureBatch& b){return tensor(b.values,{static_cast<py::ssize_t>(b.size()),16,32,32});}).def_readonly("frame_ids",&FeatureBatch::frame_ids);
 py::class_<FeatherHubertExtractor>(m,"FeatherHubertExtractor")
 .def(py::init([](std::string model,RuntimeOptions options){py::gil_scoped_release g;return std::make_unique<FeatherHubertExtractor>(model,options);}),py::arg("model"),py::arg("options")=RuntimeOptions{})
 .def("push_pcm",[](FeatherHubertExtractor& e,const py::array& a,bool final){if(a.ndim()!=1||a.size()>4*1024*1024)throw py::value_error("Expected at most 4M mono PCM16 samples");auto pcm=copy<std::int16_t>(a);py::gil_scoped_release g;return e.PushPcm(pcm.data(),pcm.size(),final);},py::arg("pcm"),py::arg("final")=false)
 .def("reset",&FeatherHubertExtractor::Reset);
 py::class_<UnetLipGenerator>(m,"UnetLipGenerator")
 .def(py::init([](std::string model,RuntimeOptions options){py::gil_scoped_release g;return std::make_unique<UnetLipGenerator>(model,options);}),py::arg("model"),py::arg("options")=RuntimeOptions{})
 .def("infer",[](UnetLipGenerator& e,const py::array& im,const py::array& au){if(!shape(im,{1,6,160,160})||!shape(au,{1,16,32,32}))throw py::value_error("Invalid UNet input shapes");auto i=copy<float>(im),a=copy<float>(au);std::vector<float> out;{py::gil_scoped_release g;out=e.Infer(i,a);}return tensor(out,{1,3,160,160});});
 py::class_<PreparedFrame>(m,"PreparedFrame")
 .def(py::init([](const py::array& bgr,const py::array& landmarks,RenderOptions o){auto im=image(bgr);auto pts=points(landmarks);py::gil_scoped_release g;return std::make_unique<PreparedFrame>(im,pts,o);}),py::arg("bgr"),py::arg("landmarks"),py::arg("options")=RenderOptions{})
 .def_property_readonly("image_tensor",[](const PreparedFrame& f){return tensor(f.ImageTensor(),{1,6,160,160});})
 .def_property_readonly("face_rect",[](const PreparedFrame& f){auto r=f.FaceRect();return py::make_tuple(r.x,r.y,r.width,r.height);})
 .def("set_frame",[](PreparedFrame& f,const py::array& bgr,const py::array& landmarks,RenderOptions o){auto im=image(bgr);auto pts=points(landmarks);py::gil_scoped_release g;f.SetFrame(im,pts,o);},py::arg("bgr"),py::arg("landmarks"),py::arg("options")=RenderOptions{})
 .def("compose",[](PreparedFrame& f,const py::array& a){if(!shape(a,{1,3,160,160}))throw py::value_error("Expected output [1,3,160,160]");auto data=copy<float>(a);cv::Mat out;{py::gil_scoped_release g;out=f.Compose(data);}py::array_t<std::uint8_t> result({out.rows,out.cols,3});std::memcpy(result.mutable_data(),out.data,out.total()*3);return result;})
 .def("reset",&PreparedFrame::Reset);
 m.def("runtime_version",&RuntimeVersion);
 m.attr("__version__")=FEATHERTALK_VERSION;
}
