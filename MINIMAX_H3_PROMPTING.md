# Writing prompts for P-H3-Image

The plugin sends your request to the model exactly as typed: the Qwen3.5-9B prompt upsampler that the
model card recommends was removed (it cost 11-17 s per edit, never ran for `generate`, and rewrote the
medium into a photograph). Everything below is what replaces it, measured on this rig.

Read this before writing a request by hand, and before letting another agent write one for you.

## The five slots of a request that works

| Slot | What it answers | Example |
|---|---|---|
| Action | what changes | `Put the girl on the canvas in the pose of <layer:ref>` |
| Pose geometry | how the body is arranged | `knees on the floor, hips raised high, chest down, legs spread wide` |
| Wardrobe / traits | which traits stay and from which picture | `keep from <Picture 1> her yellow jacket, red knee-high socks` |
| Style | which medium wins | `same anime line art and flat cel shading as the current image` |
| Exclusions | what the reference must not lend | `take only its pose, framing and camera angle; ignore its photographic style` |

A request missing the first slot is the most common failure: constraints alone produce no edit.

## Rules

1. **Open with the action verb and the target.** `Put X in the pose of Y`, `Replace the background
   with Z`, `Turn the camera to a low angle`. A prompt made only of constraints ("draw everything in
   the same style, don't copy `<Picture 2>`") re-styles an already-styled canvas and the result looks
   untouched.
2. **Assign every trait to a named picture.** `<Picture 1>` (or "the current image") is the canvas;
   `<layer:Name>` is a reference layer. Unassigned traits drift toward whichever picture is more
   salient, and a photographic reference will lend its lighting and wardrobe.
3. **Spell the pose out as geometry, not as a verb.** `one thigh crossed over the other knee, her calf
   hanging down` seated her correctly where `one leg crossed over the other` left both legs extended.
   Name joint contact, direction and what touches what.
4. **Say the medium when a photo is connected.** `same anime illustration style, line art and flat cel
   shading` keeps the 2D look; without it the pose photo's studio lighting wins and the character comes
   out as a figurine.
5. **One character on the canvas.** If two images are pasted side by side, the model edits a scene:
   "the woman on the left, the girl on the right". References belong in Reference layers, never
   composited next to the subject.
6. **100-300 characters is the sweet spot.** No shot lists, no soundscape, no camera-motion notes: the
   model reads them as video structure. Add detail only where a trait was actually lost.
7. **Do not enumerate what to preserve in edit mode.** The plugin prepends a guide that keeps
   `<Picture 1>` in charge; a list of "identity, pose, framing, background" fights every edit that
   changes one of them.
8. **One change per job when identity matters.** Pose + background + outfit in one pass splits the
   model's attention and usually lands one of the three.

## Templates

**Pose transfer (edit / refine, canvas = character).**
```
Put the girl on the canvas in the exact pose and framing of <layer:ref>: she is sitting on an office
chair with her legs open, low camera angle, full body. Keep her yellow jacket, her red knee-high socks
and her dog ears. Draw everything in the art style of the current image: the same anime line art and
flat cel shading. Take only the pose, framing and camera angle from <layer:ref>; do not copy its
photographic style or lighting.
```

**Generate from references (no canvas).** The first Reference layer becomes `<Picture 1>`; the visible
bitmap is never sent.
```
Put this girl in the pose of the girl of <Picture 2>: sat in a chair with legs open, low camera angle,
white panties visible, full body. Keep from <Picture 1> her yellow jacket, red knee-high socks, white
sneakers, dog ears and tail. Draw everything in the art style of <Picture 1>. Take only the pose,
framing and camera angle from <Picture 2>.
```

**Background swap.** Do not ask for it in the same pass as the character edit: compose the new plate
into the canvas first, then let the edit change only the character. A single pass leaves the old frame
(a neon scene inside a white border).
```
Keep the girl exactly as she is and change only her pose to the one in <layer:ref>. Keep the nightclub
background and its neon lighting from the current image unchanged.
```

**Style transfer.** Name the traits, not the vibe.
```
Redraw the girl in the art style of <layer:ref>: the same line weight, the same flat cel shading, the
same palette, the same clean background. Keep her design and her pose; change nothing else.
```

## Anti-patterns, with what they actually did

| Request | Result |
|---|---|
| Only style constraints, no action verb | Output identical to the canvas; the model had nothing to change |
| `legs crossed` | Both legs extended; wording too vague for the model to place the joints |
| Two images on the canvas + one reference | Rewrite described two subjects and converted the photo to anime |
| Photo reference assigned the style | Figurine/3D look, studio lighting, pink boots from the photo |
| Pose + new room in one pass | New room drawn inside the old canvas frame (white border) |
| `white panties` without stating it in the wardrobe list | Light grey panties inherited from the reference |

## Checklist before submitting

- [ ] The request starts with what changes.
- [ ] Every trait that must survive is assigned to `<Picture 1>` or `the current image`.
- [ ] The pose is geometry: contact, direction, what touches what.
- [ ] The medium is stated when a photo reference is connected.
- [ ] One character on the canvas; references only as Reference layers.
- [ ] Under ~300 characters, no shot list.

## Checklist after the job (agents: do not skip this)

1. Read `GET /history` and print `H3ImagePrepare.inputs`: `mode` must be `reference_edit (REF2VA)`,
   the reference slot must be wired, and `prompt` must contain your text with the plugin's guide in
   front and nothing else.
2. Check each requested trait separately — pose, wardrobe, style, framing — and report the ones that
   did not arrive instead of a verdict.
3. Measure instead of trusting a caption: mean absolute distance to the canvas and to the reference
   (a 256 px thumbnail is enough), and near-white row/column counts to catch a frame or a paste.
4. Quote the real time from the history entry; 1024² reference edits run 22-45 s on a 3090 and cost
   23.8 GB, so 4 MP with references is not available.

## Reference timings (RTX 3090, `h3ErosMax`, 1024², 8-step distilled agenda)

| Job | Time |
|---|---|
| generate, 2 references | 32-55 s |
| refine/edit, 1 reference | 22-45 s |
| same, 24 steps | 32 s |
| Qwen 2.1 edit, 20 steps, CFG 2.5 | 35-38 s |
