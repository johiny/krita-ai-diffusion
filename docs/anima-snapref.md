# Anima References with SnapRef

## Context

Anima (`Arch.anima`, the 2.9B anime model) has no native image conditioning: it only reads text and
LoRAs. [SnapRef](https://github.com/ArcAnima/SnapRef) adds references to it with a LoRA adapter that
keeps the K/V of an extra "reference" pass and injects them into the attention of every block, so the
sampler sees the reference without it being part of the prompt or the token sequence.

That makes SnapRef a sidecar to the diffusion stack, in the ControlNet / IP-Adapter family, and not a
reference mode of the model itself. The differences matter when writing prompts and when combining it
with other reference mechanisms.

## Requirements

- The ComfyUI custom node package `snapref` (node `SnapRef`, route `GET /snapref/adapters`).
- At least one SnapRef adapter in the server's loras, e.g. `snapref_v2s_pre1_fp8.safetensors`.
  The adapters are LoRAs; the plugin detects the feature by an advertised lora whose name contains
  `snapref`.

The plugin asks the server which modes each adapter can read and keeps the widest one, so no adapter
selector is shown: an adapter with a style branch (`v2s_pre1`) also serves `Character & Style` and
`Character`, while `v2a` only serves those two. Without adapter metadata every mode is offered.

## Modes and prompts

The mode is not a strength knob: it changes how the reference is encoded (its own mode embedding and
adapter branches), so it changes what the model is allowed to take from the picture.

| Mode | Reference gives | The prompt describes |
|---|---|---|
| `Character & Style` (`cs`) | character and style | only the scene: no name, no hair/eyes/clothes, no artist tags |
| `Character` (`char`) | only the character | scene plus style/artist tags |
| `Style` (`style`) | only the style | character and scene in full |

A vague prompt in `Character & Style` copies the whole picture, background included. In `Style`,
raising the weight past 1.0 starts leaking the character from the reference.

## Reference layers

Each `Reference` control layer contributes one picture with its own strength:

- **1 to 3 layers**: one `SnapRef` node, `image` / `image_2` / `image_3` with
  `weight_1` / `weight_2` / `weight_3` = the layer strength.
- **4 or more layers**: the pictures are packed side by side with `ImageStitch` into one reference,
  the way the author feeds a multi-view reference. A packed picture is a single reference for the
  node, so the per-layer weight is lost and the packed reference uses weight 1.
- **Strength 0** removes a layer from the graph entirely, like the node does.

All references of a generation share one role: their K/V are concatenated at the same RoPE frame, so
the model cannot tell "this is the character, that is the style". Reference layers must therefore
agree on the mode, and mixing modes blocks generation with a message.

Not supported, by design of the node:

- A character reference and a style reference in the same generation. Use one mode plus prompt tags,
  or two passes.
- More than three references with an individual weight. `SnapRefSimple` reads up to eight pictures
  with a strength each, but its references are files in the ComfyUI input folder, not images from the
  graph.

Two characters are one reference, as the author does it: `ImageStitch` side by side plus `2girls`
and the left/right position in the prompt. Horizontal pictures score better.

## Recommended settings

- CFG **4.5**
- `euler` (or `er_sde`) with `simple`
- **28–30 steps**
- Fidelity **1.0**, weight **1.0** to start
- The reference is shrunk to 1 MP inside the node (`ref_megapixels`), so sending a large picture
  costs upload time but not VRAM

## Implementation notes

- `apply_anima_references` (`backend/workflow.py`) collects the reference layers and builds one node;
  it runs at every call site that already applies the IP-Adapter.
- `apply_snapref` (`backend/comfy_workflow.py`) writes the node inputs.
- The `Clip` named tuple returned by `load_checkpoint_with_lora` must be unwrapped before it reaches
  the node: `Clip(model=…, arch=…)` is not JSON serializable, and the failure surfaces as
  `Server error: Object of type Arch is not JSON serializable`.
- `_update_is_supported` (`model/control.py`) owns the messages: missing adapter, adapter that cannot
  read the chosen mode, and reference layers with conflicting modes.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `The server is missing the SnapRef adapter for Anima references` | No lora with `snapref` in the name, or the `snapref` package is not installed |
| `The SnapRef adapter cannot read the reference this way: <adapter>` | The chosen adapter has no branch for that mode (`v2a` has no style) |
| `Reference layers must all use the same Reference mode` | Two reference layers with different modes |
| `The server cannot pack more than three references: ImageStitch` | More than three references on a ComfyUI build without the core `ImageStitch` node |
| `Server error: Object of type Arch is not JSON serializable` | A `Clip` reached the node inputs |
| Reference ignored, graph has one `SnapRef` | Every layer has strength 0, or the model is not Anima |

The ComfyUI console prints one line per run with the adapter, each reference size and weight, the
mode and the strength, which is the quickest way to confirm what the node received.
