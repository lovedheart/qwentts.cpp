# ROCm (HIP) vs Vulkan backend benchmark

Date: 2026-10-02. Hardware: AMD Ryzen 7 PRO 7840U, iGPU Radeon 780M
(gfx1103, 15 CU), ROCm 7.x (`/opt/rocm`), RADV driver for the Vulkan side.

## Build

- HIP backend: `./buildhip.sh` (`-DGGML_HIP=ON -DGGML_VULKAN=OFF
  -DCMAKE_HIP_ARCHITECTURES=gfx1103`), ggml 40e16e4a. Builds clean, no
  source changes needed; the backend registers as `ROCm0` and is selected
  at runtime with `GGML_BACKEND=ROCm0`.
- Vulkan backend: `./buildvulkan.sh` (`-DGGML_VULKAN=ON`). Device 0 is the
  RADV PHOENIX iGPU (`uma: 1`, coopmat capable), which is what the
  `qwen-tts.service` production unit uses via `GGML_BACKEND=Vulkan0`.

## Performance (models/qwen-talker-0.6b-base-Q8_0 + tokenizer-12hz-Q8_0)

100-word English text, same binary pair, median of runs:

| mode            | Vulkan0 wall | ROCm0 wall | delta        |
|-----------------|--------------|------------|--------------|
| default (temp)  | 16.3 s       | 21.3 s     | ROCm +31 %   |
| `--greedy`      | 92.3 s       | 119.5 s    | ROCm +30 %   |

Per-run variance is low (< 5 %) on both backends.

## Output fidelity

Same-seed and greedy runs were compared STFT-magnitude cosine:

- Vulkan vs Vulkan repeat (control): 1.0000
- Vulkan vs ROCm, same seed: 0.74; greedy: 0.75

The two backends are not bit-compatible: tiny numeric differences in the
matmul/attention kernels flip argmax/sampling decisions and the Talker
autoregression diverges into a different (still intelligible) rendition of
the same text. Duration matches to the sample; audio quality is on par
subjectively. This is the same effect the CUDA-vs-Vulkan logs in `tests/`
show (cross-backend `CodesFull exact` is only a few percent).

## Verdict

On gfx1103 the RADV Vulkan path is roughly 30 % faster than the HIP/ROCm
path for this workload (small-batch autoregressive decode, FP16 coopmat).
Keep the production service on Vulkan0. The ROCm backend is a build option
only (`buildhip.sh`), useful as a fallback where RADV is unavailable.
