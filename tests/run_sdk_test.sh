#!/usr/bin/env bash
set -euo pipefail

module=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$module"
build_root=${FEATHERTALK_TEST_BUILD_ROOT:-$module/build/sdk-tests}
jobs=${CMAKE_BUILD_PARALLEL_LEVEL:-2}
python=${PYTHON:-python3}

# A host job can supply release headers separately from the system include path.
cmake_args=()
if [[ -n ${ONNXRUNTIME_INCLUDE_DIR:-} ]]; then
    cmake_args+=("-DONNXRUNTIME_INCLUDE_DIR=$ONNXRUNTIME_INCLUDE_DIR")
fi

case ${1:-} in
    core)
        build=$build_root/core
        cmake -S "$module" -B "$build" "${cmake_args[@]}" \
            -DBUILD_TESTING=ON -DBUILD_PYTHON_BINDINGS=OFF
        cmake --build "$build" --target feathertalk_core_test --parallel "$jobs"
        ctest --test-dir "$build" --no-tests=error --output-on-failure
        ;;
    python)
        "$python" -m unittest discover -s tests/python -p 'test_*.py' -v
        ;;
    installed)
        build=$build_root/installed
        cmake -S "$module" -B "$build" "${cmake_args[@]}" \
            -DBUILD_TESTING=OFF -DBUILD_PYTHON_BINDINGS=ON
        cmake --build "$build" --parallel "$jobs"
        scratch=$(mktemp -d)
        trap 'rm -rf -- "$scratch"' EXIT
        cmake --install "$build" --prefix "$scratch/staging"
        package=$(find "$scratch/staging" -type d -name spacemit_feathertalk -print -quit)
        test -n "$package"
        cd "$scratch"
        env -u LD_LIBRARY_PATH PYTHONPATH="${package%/*}" "$python" -c 'from spacemit_feathertalk import RuntimeOptions, FeatherHubertExtractor, UnetLipGenerator; assert RuntimeOptions().provider == "cpu"'
        status=0
        env -u LD_LIBRARY_PATH "$scratch/staging/bin/feathertalk_worker" >worker.log 2>&1 || status=$?
        cat worker.log
        test "$status" -eq 2
        grep -F 'Usage: feathertalk_worker' worker.log
        ;;
    parity)
        : "${FEATHERTALK_TEST_CPU_ORT:?set the matched CPU ORT library path}"
        : "${FEATHERTALK_TEST_HUBERT:?set the FeatherHuBERT model path}"
        : "${FEATHERTALK_TEST_UNET:?set the UNet model path}"
        build=$build_root/parity
        cmake -S "$module" -B "$build" "${cmake_args[@]}" \
            -DBUILD_TESTING=OFF -DBUILD_PYTHON_BINDINGS=ON
        cmake --build "$build" --target _feathertalk --parallel "$jobs"
        export PYTHONPATH="$build${PYTHONPATH:+:$PYTHONPATH}"
        "$python" tests/python/native_parity.py --mode feature \
            --model "$FEATHERTALK_TEST_HUBERT" --ort-library "$FEATHERTALK_TEST_CPU_ORT"
        "$python" tests/python/native_parity.py --mode unet \
            --model "$FEATHERTALK_TEST_UNET" --ort-library "$FEATHERTALK_TEST_CPU_ORT"
        if [[ -n ${FEATHERTALK_TEST_UNET_ALT:-} ]]; then
            "$python" tests/python/native_parity.py --mode unet \
                --model "$FEATHERTALK_TEST_UNET_ALT" --ort-library "$FEATHERTALK_TEST_CPU_ORT"
        fi
        ;;
    *)
        echo "Usage: $0 {core|python|installed|parity}" >&2
        exit 2
        ;;
esac
