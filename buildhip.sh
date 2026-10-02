#!/bin/bash

rm -rf buildhip
mkdir buildhip
cd buildhip

# Target the local AMD iGPU/dGPU (e.g. gfx1103); rocminfo needs /dev/kfd access.
GFX="$(rocminfo 2>/dev/null | awk '/^ *Name: *gfx/{print $2; exit}')"
[ -z "$GFX" ] && GFX=gfx1103

cmake .. -DGGML_HIP=ON -DGGML_VULKAN=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES="$GFX"
cmake --build . --config Release -j "$(nproc)"
