# MiniMax H3 / P-H3 Image

MiniMax H3 is an omni-modal diffusion transformer. `p-h3-image` is the distilled image variant, and
the same image graph runs any H3 checkpoint as base model (DaSiWa, ErosMax, ref2va). The plugin
treats the whole family as one architecture: `MiniMax H3`.

ComfyUI reports `base_model: "unknown"` for every one of these files (the tooling-nodes pack does
not know the `MiniMaxH3` class unless patched), so the architecture is resolved from the filename
(`h3` / `minimax`) and, when available, from the `minimax-h3` value the patched server reports.

## Recommended settings (the recipe the model ships with)

| Knob | Value |
|---|---|
| Sampler | `er_sde` |
| Scheduler | `manual` (distilled 8-step schedule) |
| Steps | 8 |
| CFG | 1.0 |
| Negative prompt | unused at CFG 1.0 |

Everything else keeps the model in familiar territory: no LoRA required, no ControlNet, no
regional prompts and no masked inpaint. A selection (Generate and Refine alike, live preview
included) is emulated in three steps:

1. The painted area of the selection mask is measured and cropped with 25% extra context
   (minimum 32 px, aligned to 32), because the model regenerates the whole crop and needs to see
   what surrounds the edit. The mask image itself spans the whole canvas, so the selection is
   measured from its pixels, not from its bounds.
2. The crop is sent as reference image 1 and regenerated.
3. The patch is pasted back into the untouched image at the selection offset, then only the pixels
   inside the selection are composited, faded by the Selection Blend value. Selection Feather and
   Grow describe a denoise mask and are not used: pasting invented pixels into that halo is what
   made results appear to leak outside the selection.

Inside the selection everything is regenerated - the node has no mask input - so keep selections
small and prompts focused on the change when the original has to survive.

When the **H3 Image Studio** pack is installed, edits go through `H3ImagePrepare` in
`image_to_image (FL2VA)` mode instead: the latent starts from the source image with
`preserve_strength` 0.6, so the model keeps the composition instead of re-rendering the crop from
scratch. Measured on a 640x544 canvas with a 400x360 selection: no global shift (dx=0, dy=0) and
0 px changed outside the selection. Without the pack the workflow falls back to
`PH3ImageConditioning`, which only sees the crop as a reference image.

## Following Pruna's recommendations

The model card is explicit about what P-H3-Image needs, and the workflow follows it:

- **Prompt upsampler, as a separate job.** Short requests give weak results, so the request goes
  through the Qwen3.5-9B upsampler before sampling (`TextGenerate`, greedy, the `<Picture 1>` chat
  template from the official edit workflow), and the expanded brief becomes the prompt.

  It runs as its own prompt, chained by the client (`expand_h3_prompt`), because the 9 GB
  upsampler and the model do not fit in 24 GB together: in one graph ComfyUI stages both sets of
  weights on disk and an edit goes from ~1 minute to over 20. Measured chained on a 3090: 14 s
  for the upsampler and 37 s for the edit at 1568x1312, versus 3 minutes for a 4 MP edit without
  it. Live previews skip it (`WorkflowInput.expand_prompt` is only set for full jobs).
- **4 MP edits.** Crops are upscaled to 4 MP (capped at 4x linear) before sampling and scaled back
  after, per *"edits work best at 4 or 8 MP"*. Expect edits to take a minute rather than seconds.
- **`euler`** with the 8-step schedule, as in the official workflows.
- **LoRAs are model only.** P-H3-Image LoRAs (`p-h3-image_lora_bf16`, 2 GB) apply at strength 1.0
  through `LoraLoaderBypassModelOnly` and turn any MiniMax-H3 FL2VA checkpoint into the P-H3-Image
  edit model, so a finetune can be used without giving up its style.

Region layers are still unsupported and raise a clear error.

Picking the preset **"MiniMax H3 - distilled 8 steps"** applies the whole table at once. New H3
styles select it automatically while they still hold the plugin default.

## Everything is configurable

The graph passes whatever the style provides, so you can experiment freely:

- **Sampler** - any sampler ComfyUI exposes. `dpmpp_2m` and `euler` produce coherent images with
  the distilled schedule; unknown names fall back to `er_sde`.
- **Scheduler** - `manual` (distilled) or any ComfyUI scheduler. Choosing another scheduler makes
  the workflow emit `BasicScheduler` with the configured step count instead of `ManualSigmas`.
- **Steps** - honoured for every scheduler except `manual`, where the 9 documented sigmas define
  the trajectory (the step slider has no effect there).
- **CFG** - values above 1.0 add a second `PH3ImageConditioning` (empty or user negative prompt)
  and a `CFGGuider`.
- **Strength** - ignored. The model has no denoise step and its latent is a paired video+audio
  tensor from `PH3ImageConditioning`, so it cannot start from a partially noised image. The slider
  stays visible but changes nothing; a selection is always fully regenerated.

Nothing blocks a combination. Be aware that the distilled model is only tuned for the recommended
row: in a 512x512 A/B test, `dpmpp_2m + karras + 12 steps` returned an abstract colour field
instead of the prompted scene, while `dpmpp_2m` with the manual schedule stayed coherent. Use the
free knobs to explore, not as a quality guarantee.

## Performance

- 512x512, one 21 GB finetune, warm model: ~14-20 s per image (`er_sde`, 8 steps).
- First load of a 21 GB checkpoint dominates (~40-90 s).
- The distilled `p-h3-image_w4a8` (12.4 GB) is the one that comfortably reaches 8 MP on a 3090;
  the finetunes are tight at that resolution.

## Performance (RTX 3090, `--fast` dynamic VRAM loading)

Measured with `h3ErosMax` (20 GB int8), `preserve_strength` 0.6, one edited image, seed fixed.
Generation time is dominated by streaming the 20 GB transformer from disk, not by pixels:

| Config | 1088x1088 | 1440x1440 | 2048x2048 |
|---|---|---|---|
| without the upsampler | 41 s | 47 s | 45 s |
| with the upsampler | 31 s | 36 s | 50 s |
| control, fresh seed | 60 s | 55 s | 64 s |

- The upsampler costs 13-17 s warm, 28 s cold, and ~2 s when ComfyUI caches the same prompt and
  image. It never runs concurrently with the model, so it does not slow sampling down.
- Every generation re-stages the model after the upsampler job, but only the first job after a
  cold disk read pays for it: 128 s once, then 45-60 s for identical work.
- Real Krita jobs at 1312x1312 with the LoRA loaded: 59.8 s and 58.2 s, plus 12.7 s and 13.6 s of
  upsampler - about 72 s per edit.
- Peak VRAM: 21-23 GB. 4 MP + the upsampler in one graph does not fit: both sets of weights end up
  streaming and an edit takes over 20 minutes, which is why the upsampler is a separate job.
