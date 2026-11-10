cmake_minimum_required(VERSION 3.18)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" AVATAR_VERSION LIMIT_COUNT 1)
project(avatar VERSION ${AVATAR_VERSION} LANGUAGES CXX)
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "Native inference currently targets Linux. Windows supports downloader/host tests only.")
endif()
# Match the SDK's use of a separate build directory while still honoring
# an explicit build type or a multi-configuration generator.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Keep runtime library directories outside the system paths (for example a
# release ONNX Runtime archive) in the installed RPATH.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
include(CTest)
include(GNUInstallDirs)
option(BUILD_PYTHON_BINDINGS "Build pybind11 extension" ON)
option(AVATAR_WHEEL_LAYOUT "Install the extension and worker into the Python package" OFF)
# envsetup.sh exports SROBOTIS_OUTPUT_ROOT for SDK builds, where `mm` also
# builds the Python wheel into ${SROBOTIS_OUTPUT_ROOT}/dist as vad and kws do.
if(DEFINED ENV{SROBOTIS_OUTPUT_ROOT} AND NOT "$ENV{SROBOTIS_OUTPUT_ROOT}" STREQUAL "")
  set(AVATAR_IN_SDK_BUILD ON)
else()
  set(AVATAR_IN_SDK_BUILD OFF)
endif()
option(AVATAR_BUILD_WHEEL "Package this build's binaries as a Python wheel" ${AVATAR_IN_SDK_BUILD})

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h PATH_SUFFIXES onnxruntime REQUIRED)
find_library(ONNXRUNTIME_LIB onnxruntime HINTS ${ONNXRUNTIME_INCLUDE_DIR}/../lib REQUIRED)
# The SpaceMIT EP ships with spacemit-onnxruntime on K3. Hosts without it build
# a CPU-only library that rejects provider "spacemit".
find_path(SPACEMIT_ORT_INCLUDE_DIR spacemit_ort_env.h HINTS ${ONNXRUNTIME_INCLUDE_DIR} PATH_SUFFIXES onnxruntime)
find_library(SPACEMIT_EP_LIB spacemit_ep HINTS ${ONNXRUNTIME_INCLUDE_DIR}/../lib)
find_path(OPENCV_INCLUDE_DIR opencv2/core.hpp PATH_SUFFIXES opencv4 REQUIRED)
foreach(component core imgproc imgcodecs)
  find_library(OPENCV_${component} opencv_${component} REQUIRED)
  list(APPEND AVATAR_OPENCV_LIBRARIES ${OPENCV_${component}})
endforeach()
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)

add_library(avatar STATIC src/runtime.cpp src/features.cpp src/image.cpp src/models.cpp src/io.cpp)
target_include_directories(avatar
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include> ${OPENCV_INCLUDE_DIR}
  PRIVATE ${ONNXRUNTIME_INCLUDE_DIR} src)
target_link_libraries(avatar PUBLIC ${AVATAR_OPENCV_LIBRARIES} ${ONNXRUNTIME_LIB} Threads::Threads)
if(SPACEMIT_ORT_INCLUDE_DIR AND SPACEMIT_EP_LIB)
  target_compile_definitions(avatar PRIVATE AVATAR_HAVE_SPACEMIT_EP)
  target_include_directories(avatar PRIVATE ${SPACEMIT_ORT_INCLUDE_DIR})
  target_link_libraries(avatar PUBLIC ${SPACEMIT_EP_LIB})
  message(STATUS "SpaceMIT EP enabled: ${SPACEMIT_EP_LIB}")
else()
  message(WARNING "SpaceMIT EP not found; building a CPU-only avatar library")
endif()
target_compile_options(avatar PRIVATE -Wall -Wextra -Wpedantic)

add_executable(avatar_worker src/cli/worker.cpp)
target_include_directories(avatar_worker PRIVATE src ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(avatar_worker PRIVATE avatar yaml-cpp)
add_executable(avatar_file_demo examples/avatar_file_demo.cpp)
target_link_libraries(avatar_file_demo PRIVATE avatar)

if(AVATAR_WHEEL_LAYOUT)
  install(TARGETS avatar_worker RUNTIME DESTINATION spacemit_avatar/bin)
else()
  install(TARGETS avatar ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(FILES include/avatar_service.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(TARGETS avatar_worker avatar_file_demo RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(BUILD_PYTHON_BINDINGS)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_spacemit_avatar python/avatar_bindings.cpp)
  target_compile_definitions(_spacemit_avatar PRIVATE AVATAR_VERSION="${PROJECT_VERSION}")
  target_link_libraries(_spacemit_avatar PRIVATE avatar)
  if(AVATAR_WHEEL_LAYOUT)
    set(AVATAR_PYTHON_DEST spacemit_avatar)
  else()
    set(AVATAR_PYTHON_INSTALL_DIR
        "${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
        CACHE STRING "Python install directory relative to the SDK install prefix")
    if(IS_ABSOLUTE "${AVATAR_PYTHON_INSTALL_DIR}")
      message(FATAL_ERROR "AVATAR_PYTHON_INSTALL_DIR must be relative to the install prefix")
    endif()
    set(AVATAR_PYTHON_DEST "${AVATAR_PYTHON_INSTALL_DIR}/spacemit_avatar")
    install(FILES python/spacemit_avatar/__init__.py python/spacemit_avatar/pipeline.py
            DESTINATION ${AVATAR_PYTHON_DEST})
  endif()
  install(TARGETS _spacemit_avatar LIBRARY DESTINATION ${AVATAR_PYTHON_DEST})
  if(AVATAR_BUILD_WHEEL AND NOT AVATAR_WHEEL_LAYOUT)
    # Package the extension and avatar_worker built above instead of compiling
    # them again; python/setup.py copies them from AVATAR_PREBUILT_DIR.
    set(AVATAR_WHEEL_ROOT "${CMAKE_BINARY_DIR}/python_wheel")
    set(AVATAR_WHEEL_SOURCE "${AVATAR_WHEEL_ROOT}/src")
    set(AVATAR_WHEEL_PREBUILT "${AVATAR_WHEEL_ROOT}/prebuilt")
    set(AVATAR_WHEEL_OUTPUT "${AVATAR_WHEEL_ROOT}/dist")
    set(AVATAR_WHEEL_STAMP "${AVATAR_WHEEL_ROOT}/wheel.stamp")
    if(AVATAR_IN_SDK_BUILD)
      set(AVATAR_DIST_INSTALL_DIR "$ENV{SROBOTIS_OUTPUT_ROOT}/dist")
    else()
      get_filename_component(AVATAR_DIST_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/../dist" ABSOLUTE)
    endif()
    set(AVATAR_WHEEL_ROOT_INPUTS ${CMAKE_CURRENT_SOURCE_DIR}/VERSION ${CMAKE_CURRENT_SOURCE_DIR}/README.md)
    set(AVATAR_WHEEL_PYTHON_INPUTS ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.py ${CMAKE_CURRENT_SOURCE_DIR}/python/pyproject.toml)
    add_custom_command(
      OUTPUT ${AVATAR_WHEEL_STAMP}
      COMMAND ${CMAKE_COMMAND} -E rm -rf ${AVATAR_WHEEL_SOURCE} ${AVATAR_WHEEL_PREBUILT} ${AVATAR_WHEEL_OUTPUT}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${AVATAR_WHEEL_SOURCE}/python ${AVATAR_WHEEL_PREBUILT}/bin ${AVATAR_WHEEL_OUTPUT}
      COMMAND ${CMAKE_COMMAND} -E copy ${AVATAR_WHEEL_ROOT_INPUTS} ${AVATAR_WHEEL_SOURCE}
      COMMAND ${CMAKE_COMMAND} -E copy ${AVATAR_WHEEL_PYTHON_INPUTS} ${AVATAR_WHEEL_SOURCE}/python
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/python/spacemit_avatar
              ${AVATAR_WHEEL_SOURCE}/python/spacemit_avatar
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_spacemit_avatar> ${AVATAR_WHEEL_PREBUILT}
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:avatar_worker> ${AVATAR_WHEEL_PREBUILT}/bin
      COMMAND ${CMAKE_COMMAND} -E env AVATAR_PREBUILT_DIR=${AVATAR_WHEEL_PREBUILT}
              ${Python3_EXECUTABLE} -m build --wheel --no-isolation --outdir ${AVATAR_WHEEL_OUTPUT}
              ${AVATAR_WHEEL_SOURCE}/python
      COMMAND ${CMAKE_COMMAND} -E touch ${AVATAR_WHEEL_STAMP}
      DEPENDS _spacemit_avatar avatar_worker ${AVATAR_WHEEL_ROOT_INPUTS} ${AVATAR_WHEEL_PYTHON_INPUTS}
              ${CMAKE_CURRENT_SOURCE_DIR}/python/spacemit_avatar/__init__.py
              ${CMAKE_CURRENT_SOURCE_DIR}/python/spacemit_avatar/pipeline.py
      VERBATIM
      COMMENT "Building spacemit_avatar wheel into ${AVATAR_WHEEL_OUTPUT}")
    add_custom_target(avatar-python-wheel ALL DEPENDS ${AVATAR_WHEEL_STAMP})
    install(DIRECTORY ${AVATAR_WHEEL_OUTPUT}/ DESTINATION ${AVATAR_DIST_INSTALL_DIR}
            FILES_MATCHING PATTERN "*.whl")
  endif()
endif()
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
