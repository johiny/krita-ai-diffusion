# NVIDIA RTX VSR — quality modes and strength

Reference for the "NVIDIA RTX VSR" upscaler in the Krita AI Diffusion Upscale workspace.
Source of the mode semantics: `nvvfx.QualityLevel` docstrings from **nvidia-vfx 0.2.0.0**
(the Python bindings of the NVIDIA Video Effects SDK).

## Where it comes from

ComfyUI has **no RTX node in its core**. The node is provided by the official first-party
extension [`Comfy-Org/ComfyUI_Nvidia_VFX_Nodes`](https://github.com/Comfy-Org/ComfyUI_Nvidia_VFX_Nodes)
(`custom_nodes/nvidia_vfx_nodes`, node id `RTXVideoSuperResolution`). The plugin does not bundle it:
if the pack is missing, the "NVIDIA RTX VSR" entry simply does not show up in the model list.

The plugin reads the node schema from `/object_info` when it connects, so the quality list and the
optional inputs follow whatever pack version is installed:

| Pack | `nvidia-vfx` | VSR inputs | Quality modes |
|------|--------------|------------|---------------|
| 0.1.x | 0.1.x | `images`, `resize_type`, `quality` | 4 (`LOW`…`ULTRA`) |
| 0.2.x | ≥ 0.2 | + `strength`, `image_encoding` | 19 (all families below) |

## Mode families

Every family has the same intensity ladder: **LOW → ULTRA trades speed for quality**.

### Standard upscaling — `LOW`, `MEDIUM`, `HIGH`, `ULTRA`

AI upscaling trained on typical **compressed video**. These models remove compression artifacts
(macro-blocking, mosquito noise) while enlarging. `HIGH` is the SDK default.

`BICUBIC` sits at the bottom of the list: no AI at all, plain bicubic interpolation — the fastest
baseline, useful only to compare against.

Downside for clean sources: artifact suppression also eats fine texture, so on freshly rendered or
painted art these modes can look softer than expected.

### High bitrate upscaling — `HIGHBITRATE_LOW`, `HIGHBITRATE_MEDIUM`, `HIGHBITRATE_HIGH`, `HIGHBITRATE_ULTRA`

Same AI upscaling, but **artifact suppression is skipped** so already-clean detail is not degraded.
Intended for high-bitrate or lossless sources (ProRes, high-quality H.265) and, in practice, the best
family for **digital illustration and animation frames**.

### Denoise — `DENOISE_LOW` … `DENOISE_ULTRA`

**Same-resolution processing** (`output == input`). Removes noise and compression artifacts while
keeping the resolution. Higher levels remove more but can soften fine texture.

### Deblur — `DEBLUR_LOW` … `DEBLUR_ULTRA`

**Same-resolution processing.** Sharpens soft, out-of-focus or motion-blurred sources instead of
targeting noise. Aggressive levels can add halos on high-contrast line art.

### Streaming — `STREAMING_MEDIUM`, `STREAMING_ULTRA`

Tuned for broadcast and webcam streaming. They bring nothing useful to a still image job.

## RTX Strength

Float `0.0`–`1.0`, default `1.0`, forwarded to the SDK as the effect strength: how much of the
processed result is blended in. `1.0` is the full effect; lower values soften it and are handy when
a preset over-sharpens or "crunchy" the result. It only appears in the UI when the installed node
declares the input (pack 0.2.x and newer).

`image_encoding` is left at its default (`8-bit RGB`). The 10-bit option only matters for HDR or
banding-sensitive work, which this workspace does not do.

## How the workspace applies it

- The scale factor is `1.0×`–`4.0×`. VSR itself caps at 4×; the node is always asked for
  `min(factor, 4.0)` and the result is resampled (Lanczos) afterwards when the target needs it.
- VSR **rounds output dimensions to multiples of 8**. When the requested target is not a multiple of
  8, the plugin adds an `ImageScale` node to land exactly on `extent × factor`.
- **Denoise and Deblur keep the input size.** With a factor above `1.0` the plugin therefore reaches
  the target with a plain Lanczos resize — the enlargement is no longer AI. Use them at `1.0×` as a
  cleanup pass, then run a separate upscale pass.
- Non-RTX upscalers (ESRGAN-style `.pth`/`.safetensors` models) keep their usual path:
  `UpscaleModelLoader` + `ImageUpscaleWithModel`, plus a resize when the model's native factor
  (usually 4×) differs from the requested one.

## Recipes

| Input | Preset | Notes |
|-------|--------|-------|
| Clean illustration / PNG export | `HIGHBITRATE_ULTRA` | Strength `1.0`; drop to `0.7`–`0.8` if it over-sharpens |
| Reference art that went through JPEG or a lossy upload | `ULTRA` | Standard family cleans compression artifacts |
| Soft, out-of-focus or upscaled-from-small art | `DEBLUR_MEDIUM`/`DEBLUR_HIGH` at `1.0×`, then a high-bitrate upscale | Two passes |
| Scans or photos with sensor noise | `DENOISE_MEDIUM` at `1.0×`, then a high-bitrate upscale | Two passes |
| Quick draft, no GPU time to spare | `BICUBIC` or `LOW` | Baseline comparison |
| Animated frames / video pipeline | `HIGHBITRATE_*` | `STREAMING_*` only for live sources |

## Troubleshooting

- **No "NVIDIA RTX VSR" entry**: the node pack is not installed on the server, or ComfyUI did not
  reload after installing it. Check `/object_info/RTXVideoSuperResolution` in the ComfyUI web UI
  (`/object_info` is a GET endpoint; a `{}` answer means the node is absent).
- **Node present but fails to execute**: `nvidia-vfx` missing from the interpreter that runs
  ComfyUI, or a driver/CUDA mismatch. `nvidia-vfx` must be installed with the same Python that
  launches ComfyUI, and the pack requires CUDA-enabled PyTorch (cu130 or newer for 0.2.x).
- **Only 4 quality modes and no strength**: pack 0.1.x is installed. That is expected; the UI adapts
  automatically (no wrapper node needed).
