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

- **Prompt upsampler (off by default).** Short requests give weak results, so the model card runs
  them through the Qwen3.5-9B upsampler first (`TextGenerate`, greedy, the `<Picture 1>` chat
  template from the official edit workflow). The plumbing is in place
  (`h3_prompt_upsampler_enabled`), but on 24 GB the 9 GB upsampler and the model do not fit
  together: with ComfyUI's `--fast` dynamic loading both stage weights on disk and an edit goes
  from ~3 minutes to over 20. Enable it only where both models fit, or run it as a separate pass.
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
