/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "avatar_service.h"

namespace py = pybind11;
namespace avatar = SpacemiT::avatar;

namespace {

template <class T>
void Check(const py::array& array) {
    if (!array.dtype().is(py::dtype::of<T>()) || !(array.flags() & py::array::c_style)) {
        throw py::value_error("Expected exact native dtype and C-contiguous array");
    }
}

template <class T>
std::vector<T> Copy(const py::array& array) {
    Check<T>(array);
    // NumPy C-contiguous arrays can have an unaligned data pointer.
    std::vector<T> values(array.size());
    if (!values.empty()) {
        std::memcpy(values.data(), array.data(), values.size() * sizeof(T));
    }
    return values;
}

py::array_t<float> Tensor(const std::vector<float>& values, std::vector<py::ssize_t> shape) {
    py::array_t<float> array(shape);
    if (!values.empty()) {
        std::memcpy(array.mutable_data(), values.data(), values.size() * sizeof(float));
    }
    return array;
}

cv::Mat Image(const py::array& array) {
    Check<std::uint8_t>(array);
    if (array.ndim() != 3 || array.shape(2) != 3 || array.shape(0) < 1 || array.shape(1) < 1 ||
        array.shape(0) > 16384 || array.shape(1) > 16384) {
        throw py::value_error("Expected BGR uint8 HxWx3");
    }
    return cv::Mat(static_cast<int>(array.shape(0)), static_cast<int>(array.shape(1)), CV_8UC3,
        const_cast<void*>(array.data())).clone();
}

std::vector<cv::Point2f> Points(const py::array& array) {
    if (array.ndim() != 2 || array.shape(1) != 2 || array.shape(0) < 110 || array.shape(0) > 10000) {
        throw py::value_error("Expected float32 [N>=110,2] landmarks");
    }
    const auto values = Copy<float>(array);
    std::vector<cv::Point2f> points;
    points.reserve(array.shape(0));
    for (std::size_t i = 0; i < values.size(); i += 2) {
        points.emplace_back(values[i], values[i + 1]);
    }
    return points;
}

bool HasShape(const py::array& array, const std::vector<py::ssize_t>& dims) {
    if (array.ndim() != static_cast<py::ssize_t>(dims.size())) {
        return false;
    }
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (array.shape(i) != dims[i]) {
            return false;
        }
    }
    return true;
}

}  // namespace

PYBIND11_MODULE(_spacemit_avatar, m) {
    m.doc() = "bianbu-avatar FeatherHuBERT and person UNet inference";

    py::class_<avatar::RuntimeOptions>(m, "RuntimeOptions")
        .def(py::init<>())
        .def_readwrite("provider", &avatar::RuntimeOptions::provider)
        .def_readwrite("profile_prefix", &avatar::RuntimeOptions::profile_prefix)
        .def_readwrite("threads", &avatar::RuntimeOptions::threads)
        .def_readwrite("ep_threads", &avatar::RuntimeOptions::ep_threads);

    py::class_<avatar::RenderOptions>(m, "RenderOptions")
        .def(py::init<>())
        .def_readwrite("feather", &avatar::RenderOptions::feather)
        .def_readwrite("mouth_only_blend", &avatar::RenderOptions::mouth_only_blend)
        .def_readwrite("native_pasteback", &avatar::RenderOptions::native_pasteback)
        .def_readwrite("temporal_smooth", &avatar::RenderOptions::temporal_smooth)
        .def_readwrite("mouth_sharpen", &avatar::RenderOptions::mouth_sharpen)
        .def_readwrite("smooth_alpha", &avatar::RenderOptions::smooth_alpha)
        .def_readwrite("patch_lowpass", &avatar::RenderOptions::patch_lowpass);

    py::class_<avatar::FeatureBatch>(m, "FeatureBatch")
        .def_property_readonly("windows", [](const avatar::FeatureBatch& batch) {
            return Tensor(batch.values, {static_cast<py::ssize_t>(batch.size()), 16, 32, 32});
        })
        .def_readonly("frame_ids", &avatar::FeatureBatch::frame_ids);

    py::class_<avatar::FeatherHubertExtractor>(m, "FeatherHubertExtractor")
        .def(py::init([](const std::string& model, const avatar::RuntimeOptions& options) {
            py::gil_scoped_release release;
            return std::make_unique<avatar::FeatherHubertExtractor>(model, options);
        }), py::arg("model"), py::arg("options") = avatar::RuntimeOptions{})
        .def("push_pcm", [](avatar::FeatherHubertExtractor& extractor, const py::array& array, bool final) {
            if (array.ndim() != 1 || array.size() > 4 * 1024 * 1024) {
                throw py::value_error("Expected at most 4M mono PCM16 samples");
            }
            auto pcm = Copy<std::int16_t>(array);
            py::gil_scoped_release release;
            return extractor.PushPcm(pcm.data(), pcm.size(), final);
        }, py::arg("pcm"), py::arg("final") = false)
        .def("reset", &avatar::FeatherHubertExtractor::Reset);

    py::class_<avatar::UnetLipGenerator>(m, "UnetLipGenerator")
        .def(py::init([](const std::string& model, const avatar::RuntimeOptions& options) {
            py::gil_scoped_release release;
            return std::make_unique<avatar::UnetLipGenerator>(model, options);
        }), py::arg("model"), py::arg("options") = avatar::RuntimeOptions{})
        .def("infer", [](avatar::UnetLipGenerator& generator, const py::array& image, const py::array& audio) {
            if (!HasShape(image, {1, 6, 160, 160}) || !HasShape(audio, {1, 16, 32, 32})) {
                throw py::value_error("Invalid UNet input shapes");
            }
            auto image_values = Copy<float>(image);
            auto audio_values = Copy<float>(audio);
            std::vector<float> output;
            {
                py::gil_scoped_release release;
                output = generator.Infer(image_values, audio_values);
            }
            return Tensor(output, {1, 3, 160, 160});
        });

    py::class_<avatar::PreparedFrame>(m, "PreparedFrame")
        .def(py::init([](const py::array& bgr, const py::array& landmarks, const avatar::RenderOptions& options) {
            auto image = Image(bgr);
            auto points = Points(landmarks);
            py::gil_scoped_release release;
            return std::make_unique<avatar::PreparedFrame>(image, points, options);
        }), py::arg("bgr"), py::arg("landmarks"), py::arg("options") = avatar::RenderOptions{})
        .def_property_readonly("image_tensor", [](const avatar::PreparedFrame& frame) {
            return Tensor(frame.ImageTensor(), {1, 6, 160, 160});
        })
        .def_property_readonly("face_rect", [](const avatar::PreparedFrame& frame) {
            auto rect = frame.FaceRect();
            return py::make_tuple(rect.x, rect.y, rect.width, rect.height);
        })
        .def("set_frame", [](avatar::PreparedFrame& frame, const py::array& bgr, const py::array& landmarks,
                const avatar::RenderOptions& options) {
            auto image = Image(bgr);
            auto points = Points(landmarks);
            py::gil_scoped_release release;
            frame.SetFrame(image, points, options);
        }, py::arg("bgr"), py::arg("landmarks"), py::arg("options") = avatar::RenderOptions{})
        .def("compose", [](avatar::PreparedFrame& frame, const py::array& output) {
            if (!HasShape(output, {1, 3, 160, 160})) {
                throw py::value_error("Expected output [1,3,160,160]");
            }
            auto values = Copy<float>(output);
            cv::Mat composed;
            {
                py::gil_scoped_release release;
                composed = frame.Compose(values);
            }
            py::array_t<std::uint8_t> result({composed.rows, composed.cols, 3});
            std::memcpy(result.mutable_data(), composed.data, composed.total() * 3);
            return result;
        })
        .def("reset", &avatar::PreparedFrame::Reset);

    m.def("runtime_version", &avatar::RuntimeVersion);
    m.attr("__version__") = AVATAR_VERSION;
}
