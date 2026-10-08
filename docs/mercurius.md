# Mercurius-1-4B support (llama-mercurius fork)

This fork runs [Mercurius-1-4B](https://huggingface.co/Minerva-Laboratories/Mercurius-1-4B) GGUFs: Qwen3.5-4B rebuilt
with absorbed MLA attention, GDN-2 linear attention and a 4-bit TurboQuant KV cache.

It is llama.cpp → [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) (TurboQuant KV cache
types) → this fork, branch `mercurius`. Everything else in llama.cpp works as usual.

## Build and run

```bash
git clone -b mercurius https://github.com/alexhergomz/llama-mercurius
cd llama-mercurius
cmake -B build -DGGML_CUDA=ON          # NVIDIA GPU; omit -DGGML_CUDA=ON for a CPU-only build (slow)
cmake --build build -j --target llama-server llama-cli
./build/bin/llama-server -m mercurius-1-4b-NF4.gguf -ngl 99 -c 32768 -fa on
```

- `-fa on` selects the fast prefill path (sliced expanded attention). Without it, prefill uses the absorbed kernel.
- The cache format is fixed by the model: `-ctk` / `-ctv` have no effect. Context shifting is not supported.
- The cache must be one KV stream: llama-server's default (automatic slots, unified cache) works; with an explicit
  `-np N` (N > 1) also pass `--kv-unified`.

## Hardware

| Backend | Status |
|---|---|
| NVIDIA CUDA, Ampere and newer (RTX 30xx/40xx/50xx, A100, H100, Jetson Orin/Thor) | Supported. Tested on Jetson AGX Orin (sm_87); other GPUs not yet tested. |
| NVIDIA CUDA, Turing (RTX 20xx, T4) | Should work: kernels compile for sm_75 (plain copies replace `cp.async`); not run-tested. |
| NVIDIA Volta and older | Not supported (the sliced prefill needs Turing tensor-core MMA; current CUDA toolkits no longer target sm_70). |
| CPU | Works (every new op has a CPU implementation), but slow. |
| AMD ROCm / Moore Threads | Not supported yet: the tensor-core kernels are compiled out and the attention ops fall back to the CPU. |
| Apple Metal, Vulkan, SYCL, OpenCL | Not supported yet (no kernels for the new ops). |

## What the fork adds

- Architecture `mercurius` (`src/models/mercurius.cpp`): GDN-2 layers with channel-wise decay / erase / write gates
  (shared VeRA adapters, factored per-head bases, decay LoRA) and absorbed-MLA layers with a decoupled RoPE key.
- Weight type `NF4` (exact NormalFloat-4, the format the model was trained with): CPU, CUDA dequant, get_rows and
  vector-dot kernels; gguf-py support.
- `GGML_OP_GATED_DELTA_NET`: channel-wise erase gate (CPU, CUDA).
- Packed TurboQuant cache ops: `GGML_OP_MERC_TQ_PACK` / `_UNPACK` (rotation + Lloyd-Max codebook, fp16 norms).
- `GGML_OP_MERC_TQ_ATTN`: attention straight from the packed cache. Decode: split-KV flash decoding (running softmax
  per block, async staging, RoPE keys on tensor cores). Short batches / `-fa off` prefill: query-tiled flash kernel.
- `GGML_OP_MERC_TQ_PREFILL`: prefill in the expanded form without materializing the expanded cache. The cache is
  processed in 16k-cell slices (decode, expansion GEMMs, MMA flash attention in partial mode) merged by log-sum-exp;
  fixed workspace (~150 MB) held in the op's compute-buffer slot.
- `GGML_OP_MERC_TQ_EXPAND`: expanded K / V of a packed cache (reference / debugging).
- Flash-attention MMA configs and template instances for 448-wide keys / 256-wide values.

Tests: `./build/bin/test-backend-ops -o MERC_TQ_ATTN` (also `MERC_TQ_PREFILL`, `MERC_TQ_EXPAND`, `MERC_TQ_UNPACK`,
`GATED_DELTA_NET`, `FLASH_ATTN_EXT -p hsk=448`) compares each backend against the CPU implementation.

## Converting

GGUFs are produced from the deployed PyTorch model with `deploy/mercurius_gguf.py` in the OpenDecider repository.
Ready-made files: the `gguf/` folder of the Hugging Face repository above.
