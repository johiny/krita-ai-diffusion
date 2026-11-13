# H3 Inpaint: Lessons Learned

## Context

MiniMax H3 (ErosMax, DaSiWa, ref2va) inpaint in krita-ai-diffusion.
Tested across multiple iterations to reduce drift at 100% denoise.

## Architecture

H3 is a **video model** (36-layer attention VAE decoder). The inpaint uses
`VAEEncode` + `SetLatentNoiseMask` to pin the canvas outside the selection,
then sends the sampler output directly (no compositing).

## What works

| Fix | Effect | Status |
|---|---|---|
| `H3LatentForModel` wrapper | Prevents crash (IndexError: index 1) — H3 expects 2-stream latent (video + audio) | **Kept** |
| `apply_grow_feather` (grow + feather on generation mask) | Controls denoise area size and edge blur | **Kept** |
| No compositing (`send_image(out_image)` directly) | Eliminates drift — sampler output is already correct | **Kept** |
| `StabilizeMask` (epsilon=0.01) | Thresholds mask to binary | **Kept** |

## What doesn't work (tested and reverted)

| Fix | Effect | Status |
|---|---|---|
| Grid snap (32 px) | Rounds mask to cell grid — worsens drift | **Removed** |
| Hard generation mask (no feather) | Binary mask without feather — worsens drift | **Reverted** |
| Grid rounding (grow/feather/blend to 32) | Rounds params to cell grid — worsens drift | **Removed** |
| Compositing (`apply_mask` + `shrink(8, 16)`) | Re-mixes already-correct pixels — causes drift | **Removed** |
| Blend applied to generation mask | Can make mask empty (shrink removes all denoise area) — degenerate at certain feather values | **Removed** |

## Key findings

### 1. The custom node (H3LatentForModel) IS necessary

Without it, the model crashes:
```
IndexError: index 1 is out of bounds for dimension 0 with size 1
```
H3 expects a 2-element latent (video + audio). `VAEEncode` produces 1. The node wraps as `(video, zeros_audio)`.

### 2. Compositing causes drift in H3 (but not in Qwen 2.1)

- **Qwen 2.1**: sampler doesn't perfectly preserve canvas. Compositing (`apply_mask` + shrink) cleans up the seam. Necessary.
- **H3**: `SetLatentNoiseMask` perfectly preserves canvas. Compositing is redundant — re-applies mask on already-correct pixels, introducing drift via the 36-layer attention decoder.

### 3. Official model vs finetunes

| Model | Drift at 100% denoise | Reason |
|---|---|---|
| MiniMax-H3-Image (official) | Almost none | Trained for image editing with preservation |
| ErosMax beta7 | Notable | Video finetune, not trained for mask-based editing |
| DaSiWa Hybrid V4 | Notable | Video finetune, prioritizes motion/style over preservation |

The drift in finetunes is **inherent to training**. No code fix can compensate.

### 4. Community consensus (ethanfel, minimaxh3.cc, diffusers-modular)

- "Do not feather the mask fed to H3" — intermediate timesteps mix source and repaint
- BUT in practice, feather ON the generation mask works better than without (tested)
- The VAE decoder propagates changes: 5.5/255 within 8px, gone by 128px
- `mask_drop_step` technique: mask holds for first steps, then drops for free-refine

### 5. Blend behavior

- In standard models (Qwen, Anima): blend controls compositing transition (paste)
- In H3: blend has no safe application point (compositing causes drift, generation mask causes degeneration)
- **Blend is effectively disabled in H3**

## Effective configuration

```
in_mask = load_mask(hires_mask)           # no snap, no rounding
inpaint_mask = apply_grow_feather(...)     # grow + feather (works)
inpaint_mask = stabilize_mask(...)         # threshold to binary
latent = vae_encode(in_image)
latent = H3LatentForModel(latent)         # prevents crash
latent = set_latent_noise_mask(latent, inpaint_mask)
# NO compositing — send output directly
w.send_image(out_image)
```

## Working parameters

| Parameter | Default | Effect | Works? |
|---|---|---|---|
| Feather (10%) | Fraction of selection size | Blur of generation mask | Yes |
| Padding (6%) | Fraction of selection size | Dilation (grow) of generation mask | Yes |
| Grow Offset (4 px) | Fixed | Binary dilation before feather | Yes |
| Min Transition (32 px) | Fixed | Minimum feather in pixels | Yes |
| Blend (25 px) | Fixed | No effect in H3 | No |

## Recommendations for users

1. **Use official model** (MiniMax-H3-Image) for exact preservation
2. **Finetunes** (ErosMax, DaSiWa): use Strength 55-65% to control drift
3. **Prompts**: "keep everything else exactly the same" helps
4. **Small selections**: less 32px cells to regenerate = less drift
5. **Blend**: ignore it in H3 (no effect)

## Files changed

- `ai_diffusion/backend/workflow.py` — `inpaint_h3()` function
- `tests/test_h3_image.py` — updated tests to match new behavior
