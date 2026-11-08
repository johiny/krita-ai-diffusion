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
included) is emulated: the selection is cropped, sent as reference image 1, and composited back
through the mask. Region layers are still unsupported and raise a clear error.

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

Nothing blocks a combination. Be aware that the distilled model is only tuned for the recommended
row: in a 512x512 A/B test, `dpmpp_2m + karras + 12 steps` returned an abstract colour field
instead of the prompted scene, while `dpmpp_2m` with the manual schedule stayed coherent. Use the
free knobs to explore, not as a quality guarantee.

## Performance

- 512x512, one 21 GB finetune, warm model: ~14-20 s per image (`er_sde`, 8 steps).
- First load of a 21 GB checkpoint dominates (~40-90 s).
- The distilled `p-h3-image_w4a8` (12.4 GB) is the one that comfortably reaches 8 MP on a 3090;
  the finetunes are tight at that resolution.
