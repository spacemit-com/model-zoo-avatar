#!/usr/bin/env bash
#
# Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# Avatar component test entry shared by test.yaml, CI and local runs.
#   core           C++ contracts (ctest)
#   config         option defaults, linked runtime and shipped configs
#   invalid-input  invalid library inputs and avatar_worker command lines
#   python         Python unit tests (downloader, worker lifecycle)
#   installed      install to a temporary prefix and load the package from outside the tree
#   parity         real-model parity against the Python reference (needs model paths)

set -euo pipefail

module=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$module"
build_root=${AVATAR_TEST_BUILD_ROOT:-$module/build/sdk-tests}
jobs=${CMAKE_BUILD_PARALLEL_LEVEL:-2}
python=${PYTHON:-python3}

# A host job can supply release headers separately from the system include path.
# Test builds never package a wheel, so they leave the SDK's output/dist alone.
cmake_args=(-DAVATAR_BUILD_WHEEL=OFF)
if [[ -n ${ONNXRUNTIME_INCLUDE_DIR:-} ]]; then
    cmake_args+=("-DONNXRUNTIME_INCLUDE_DIR=$ONNXRUNTIME_INCLUDE_DIR")
fi

build_contracts() {
    build=$build_root/contracts
    cmake -S "$module" -B "$build" "${cmake_args[@]}" -DBUILD_TESTING=ON -DBUILD_PYTHON_BINDINGS=OFF
    cmake --build "$build" --target avatar_pr_contract_test avatar_worker --parallel "$jobs"
}

# avatar_worker must reject the command line with exit code 2 and the given message.
expect_worker_error() {
    local message=$1
    shift
    local status=0
    "$build/avatar_worker" "$@" >"$build/worker.log" 2>&1 || status=$?
    if [[ $status -ne 2 ]] || ! grep -qF "$message" "$build/worker.log"; then
        echo "avatar_worker $* did not fail with '$message' (exit $status):" >&2
        cat "$build/worker.log" >&2
        exit 1
    fi
}

case ${1:-} in
    core)
        build_contracts
        ctest --test-dir "$build" --no-tests=error --output-on-failure
        ;;
    config)
        build_contracts
        "$build/tests/avatar_pr_contract_test" --config-contract
        ;;
    invalid-input)
        build_contracts
        "$build/tests/avatar_pr_contract_test" --invalid-input-error-path
        scratch=$(mktemp -d)
        trap 'rm -rf -- "$scratch"' EXIT
        printf 'schema: 2\n' >"$scratch/schema2.yaml"
        expect_worker_error 'Usage: avatar_worker'
        expect_worker_error 'Unknown option' render --bogus value
        expect_worker_error 'Missing option value' render --config
        expect_worker_error 'config and output required' extract --audio a.wav
        expect_worker_error 'Expected config schema=1' extract --config "$scratch/schema2.yaml" --output out.ftf
        expect_worker_error 'Unknown mode' convert --config configs/feifan.yaml --output out
        expect_worker_error 'features required' render --config configs/feifan.yaml --output "$scratch/frames"
        echo 'avatar_worker invalid command lines: passed'
        ;;
    python)
        "$python" -m unittest discover -s tests/python -p 'test_*.py' -v
        ;;
    installed)
        build=$build_root/installed
        cmake -S "$module" -B "$build" "${cmake_args[@]}" -DBUILD_TESTING=OFF -DBUILD_PYTHON_BINDINGS=ON
        cmake --build "$build" --parallel "$jobs"
        scratch=$(mktemp -d)
        trap 'rm -rf -- "$scratch"' EXIT
        cmake --install "$build" --prefix "$scratch/staging"
        package=$(find "$scratch/staging" -type d -name spacemit_avatar -print -quit)
        test -n "$package"
        cd "$scratch"
        env -u LD_LIBRARY_PATH PYTHONPATH="${package%/*}" "$python" -c \
            'from spacemit_avatar import RuntimeOptions, FeatherHubertExtractor, UnetLipGenerator, runtime_version
assert RuntimeOptions().provider == "cpu" and runtime_version()'
        status=0
        env -u LD_LIBRARY_PATH "$scratch/staging/bin/avatar_worker" >worker.log 2>&1 || status=$?
        cat worker.log
        test "$status" -eq 2
        grep -F 'Usage: avatar_worker' worker.log
        env -u LD_LIBRARY_PATH "$scratch/staging/bin/avatar_file_demo" --help | grep -F 'Usage: avatar_file_demo'
        test -f "$scratch/staging/include/avatar_service.h"
        test -f "$scratch/staging/lib/libavatar.a"
        ;;
    parity)
        : "${AVATAR_TEST_HUBERT:?set the FeatherHuBERT model path}"
        : "${AVATAR_TEST_UNET:?set the UNet model path}"
        build=$build_root/parity
        cmake -S "$module" -B "$build" "${cmake_args[@]}" -DBUILD_TESTING=OFF -DBUILD_PYTHON_BINDINGS=ON
        cmake --build "$build" --target _spacemit_avatar --parallel "$jobs"
        export PYTHONPATH="$build${PYTHONPATH:+:$PYTHONPATH}"
        "$python" tests/python/native_parity.py --mode feature --model "$AVATAR_TEST_HUBERT"
        "$python" tests/python/native_parity.py --mode unet --model "$AVATAR_TEST_UNET"
        if [[ -n ${AVATAR_TEST_UNET_ALT:-} ]]; then
            "$python" tests/python/native_parity.py --mode unet --model "$AVATAR_TEST_UNET_ALT"
        fi
        ;;
    *)
        echo "Usage: $0 {core|config|invalid-input|python|installed|parity}" >&2
        exit 2
        ;;
esac
