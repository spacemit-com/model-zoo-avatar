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
include(CTest)
include(GNUInstallDirs)
option(BUILD_PYTHON_BINDINGS "Build pybind11 extension" ON)
option(AVATAR_WHEEL_LAYOUT "Install libraries and worker into Python package" OFF)
# envsetup.sh exports SROBOTIS_OUTPUT_ROOT for SDK builds, where `mm` also
# builds the Python wheel into ${SROBOTIS_OUTPUT_ROOT}/dist as vad and kws do.
if(DEFINED ENV{SROBOTIS_OUTPUT_ROOT} AND NOT "$ENV{SROBOTIS_OUTPUT_ROOT}" STREQUAL "")
  set(AVATAR_IN_SDK_BUILD ON)
else()
  set(AVATAR_IN_SDK_BUILD OFF)
endif()
option(AVATAR_BUILD_WHEEL "Package this build's binaries as a Python wheel" ${AVATAR_IN_SDK_BUILD})
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h PATH_SUFFIXES onnxruntime REQUIRED)
find_path(OPENCV_INCLUDE_DIR opencv2/core.hpp PATH_SUFFIXES opencv4 REQUIRED)
foreach(component core imgproc imgcodecs)
  find_library(OPENCV_${component} opencv_${component} REQUIRED)
  list(APPEND AVATAR_OPENCV_LIBRARIES ${OPENCV_${component}})
endforeach()
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)
add_library(avatar SHARED src/runtime.cpp src/features.cpp src/image.cpp src/models.cpp src/io.cpp)
target_include_directories(avatar PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include> ${OPENCV_INCLUDE_DIR} PRIVATE ${ONNXRUNTIME_INCLUDE_DIR} src)
target_compile_definitions(avatar PRIVATE ORT_API_MANUAL_INIT)
target_link_libraries(avatar PUBLIC ${AVATAR_OPENCV_LIBRARIES} PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)
target_compile_options(avatar PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(avatar PROPERTIES SOVERSION 0 INSTALL_RPATH "$ORIGIN")
add_executable(avatar_worker src/cli/worker.cpp)
target_include_directories(avatar_worker PRIVATE src)
target_link_libraries(avatar_worker PRIVATE avatar yaml-cpp)
if(AVATAR_WHEEL_LAYOUT)
  set(AVATAR_LIB_DEST spacemit_avatar)
  set(AVATAR_BIN_DEST spacemit_avatar/bin)
  set_target_properties(avatar_worker PROPERTIES INSTALL_RPATH "$ORIGIN/..")
else()
  set_target_properties(avatar PROPERTIES VERSION ${PROJECT_VERSION})
  set(AVATAR_LIB_DEST ${CMAKE_INSTALL_LIBDIR})
  set(AVATAR_BIN_DEST ${CMAKE_INSTALL_BINDIR})
  # The same binaries find libavatar in the build tree ($ORIGIN), in the wheel
  # (worker in bin/, $ORIGIN/..) and in the install prefix, so the wheel can
  # package them without relinking.
  set_target_properties(avatar avatar_worker PROPERTIES BUILD_WITH_INSTALL_RPATH ON)
  set_target_properties(avatar_worker PROPERTIES
    INSTALL_RPATH "$ORIGIN;$ORIGIN/..;$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()
if(AVATAR_WHEEL_LAYOUT)
  install(TARGETS avatar LIBRARY DESTINATION ${AVATAR_LIB_DEST} NAMELINK_SKIP)
else()
  install(TARGETS avatar LIBRARY DESTINATION ${AVATAR_LIB_DEST})
  install(FILES include/avatar_service.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()
install(TARGETS avatar_worker RUNTIME DESTINATION ${AVATAR_BIN_DEST})
if(BUILD_PYTHON_BINDINGS)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_spacemit_avatar python/avatar_bindings.cpp)
  target_compile_definitions(_spacemit_avatar PRIVATE AVATAR_VERSION="${PROJECT_VERSION}")
  target_link_libraries(_spacemit_avatar PRIVATE avatar)
  if(AVATAR_WHEEL_LAYOUT)
    set(AVATAR_PYTHON_DEST spacemit_avatar)
    set(AVATAR_PYTHON_RPATH "$ORIGIN")
  else()
    set(AVATAR_PYTHON_INSTALL_DIR
        "${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
        CACHE STRING "Python install directory relative to the SDK install prefix")
    if(IS_ABSOLUTE "${AVATAR_PYTHON_INSTALL_DIR}")
      message(FATAL_ERROR "AVATAR_PYTHON_INSTALL_DIR must be relative to the install prefix")
    endif()
    set(AVATAR_PYTHON_DEST "${AVATAR_PYTHON_INSTALL_DIR}/spacemit_avatar")
    file(RELATIVE_PATH AVATAR_PYTHON_TO_LIB "/${AVATAR_PYTHON_DEST}" "/${CMAKE_INSTALL_LIBDIR}")
    set(AVATAR_PYTHON_RPATH "$ORIGIN;$ORIGIN/${AVATAR_PYTHON_TO_LIB}")
    set_target_properties(_spacemit_avatar PROPERTIES BUILD_WITH_INSTALL_RPATH ON)
    install(FILES python/spacemit_avatar/__init__.py python/spacemit_avatar/pipeline.py
            DESTINATION ${AVATAR_PYTHON_DEST})
  endif()
  set_target_properties(_spacemit_avatar PROPERTIES INSTALL_RPATH "${AVATAR_PYTHON_RPATH}")
  install(TARGETS _spacemit_avatar LIBRARY DESTINATION ${AVATAR_PYTHON_DEST})
  if(AVATAR_BUILD_WHEEL AND NOT AVATAR_WHEEL_LAYOUT)
    # Package the extension, libavatar and avatar_worker built above instead of
    # compiling them again; setup.py copies them from AVATAR_PREBUILT_DIR.
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
    set(AVATAR_WHEEL_INPUTS
      ${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml
      ${CMAKE_CURRENT_SOURCE_DIR}/VERSION ${CMAKE_CURRENT_SOURCE_DIR}/README.md
      ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE ${CMAKE_CURRENT_SOURCE_DIR}/NOTICE)
    add_custom_command(
      OUTPUT ${AVATAR_WHEEL_STAMP}
      COMMAND ${CMAKE_COMMAND} -E rm -rf ${AVATAR_WHEEL_SOURCE} ${AVATAR_WHEEL_PREBUILT} ${AVATAR_WHEEL_OUTPUT}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${AVATAR_WHEEL_SOURCE} ${AVATAR_WHEEL_PREBUILT}/bin ${AVATAR_WHEEL_OUTPUT}
      COMMAND ${CMAKE_COMMAND} -E copy ${AVATAR_WHEEL_INPUTS} ${AVATAR_WHEEL_SOURCE}
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/python/spacemit_avatar ${AVATAR_WHEEL_SOURCE}/python/spacemit_avatar
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_spacemit_avatar> ${AVATAR_WHEEL_PREBUILT}
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:avatar> ${AVATAR_WHEEL_PREBUILT}/$<TARGET_SONAME_FILE_NAME:avatar>
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:avatar_worker> ${AVATAR_WHEEL_PREBUILT}/bin
      COMMAND ${CMAKE_COMMAND} -E env AVATAR_PREBUILT_DIR=${AVATAR_WHEEL_PREBUILT}
              ${Python3_EXECUTABLE} -m build --wheel --no-isolation --outdir ${AVATAR_WHEEL_OUTPUT} ${AVATAR_WHEEL_SOURCE}
      COMMAND ${CMAKE_COMMAND} -E touch ${AVATAR_WHEEL_STAMP}
      DEPENDS _spacemit_avatar avatar avatar_worker ${AVATAR_WHEEL_INPUTS}
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
