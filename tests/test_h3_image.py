"""MiniMax H3 support: architecture detection and prompt shape.

The family (P-H3 Image distillation and its finetunes) uses one conditioning node that builds
both the conditioning and the latent, a fixed 8-step schedule and no CFG.
"""

import asyncio
import dataclasses
import json
import math
from pathlib import Path

import pytest

from ai_diffusion.backend import resources, workflow
from ai_diffusion.backend.api import (
    CheckpointInput,
    ConditioningInput,
    ControlInput,
    ExtentInput,
    ImageInput,
    InpaintMode,
    InpaintParams,
    SamplingInput,
)
from ai_diffusion.backend.client import CheckpointInfo, ClientModels
from ai_diffusion.backend.comfy_workflow import ComfyObjectInfo, ComfyWorkflow
from ai_diffusion.backend.resolution import ScaledExtent
from ai_diffusion.files import FileFormat
from ai_diffusion.image import Bounds, Extent, Image
from ai_diffusion.style import h3_sampler_preset
from ai_diffusion.util import PluginError

CHECKPOINT = "p-h3-image_w4a8.safetensors"
TEXT_ENCODER = "p-h3-image_text_encoder_w4a8.safetensors"
VAE = "p-h3-image_vae_int8_convrot.safetensors"


def mask_with_rect(extent: Extent, rect: Bounds) -> Image:
    """Grayscale mask with a white rectangle, like the selection the UI sends."""
    from PyQt6.QtCore import QRect, Qt
    from PyQt6.QtGui import QImage, QPainter

    qimage = QImage(extent.width, extent.height, QImage.Format.Format_Grayscale8)
    qimage.fill(Qt.GlobalColor.black)
    painter = QPainter(qimage)
    painter.fillRect(QRect(rect.x, rect.y, rect.width, rect.height), Qt.GlobalColor.white)
    painter.end()
    return Image(qimage)


def node_defs() -> ComfyObjectInfo:
    nodes = {
        "UNETLoader": {"input": {"required": {"unet_name": ["COMBO", {}]}}},
        "CLIPLoader": {
            "input": {
                "required": {
                    "clip_name": ["COMBO", {}],
                    "type": ["COMBO", {}],
                    "device": ["COMBO", {"options": ["default"]}],
                }
            }
        },
        "VAELoader": {"input": {"required": {"vae_name": ["COMBO", {}]}}},
        "MiniMaxH3SigmaShift": {
            "input": {
                "required": {
                    "model": ["MODEL", {}],
                    "shift_video": ["FLOAT", {"default": 1.0}],
                    "shift_audio": ["FLOAT", {"default": 1.0}],
                }
            }
        },
        "PH3ImageConditioning": {
            "input": {
                "required": {
                    "clip": ["CLIP", {}],
                    "prompt": ["STRING", {"default": ""}],
                    "width": ["INT", {"default": 0}],
                    "height": ["INT", {"default": 0}],
                },
                "optional": {"vae": ["VAE", {}], "images": ["COMFY_AUTOGROW_V3", {}]},
            }
        },
        "BasicGuider": {"input": {"required": {}}},
        "KSamplerSelect": {
            "input": {
                "required": {
                    "sampler_name": ["COMBO", {"options": ["er_sde", "euler", "dpmpp_2m"]}]
                }
            }
        },
        "ImageBlend": {
            "input": {
                "required": {
                    "image1": ["IMAGE", {}],
                    "image2": ["IMAGE", {}],
                    "blend_factor": ["FLOAT", {"default": 0.5}],
                    "blend_mode": ["COMBO", {"options": ["normal"]}],
                }
            }
        },
        "BasicScheduler": {
            "input": {
                "required": {
                    "model": ["MODEL", {}],
                    "scheduler": ["COMBO", {"options": ["normal", "karras", "beta"]}],
                    "steps": ["INT", {"default": 20}],
                    "denoise": ["FLOAT", {"default": 1.0}],
                }
            }
        },
        "ManualSigmas": {"input": {"required": {"sigmas": ["STRING", {}]}}},
        "RandomNoise": {"input": {"required": {"noise_seed": ["INT", {"default": 0}]}}},
        "SamplerCustomAdvanced": {"input": {"required": {}}},
        "VAEDecode": {"input": {"required": {}}},
        "RepeatLatentBatch": {"input": {"required": {}}},
        "ETN_LoadImageCache": {"input": {"required": {}}},
        "UpscaleModelLoader": {
            "input": {"required": {"model_name": ["COMBO", {"options": ["4x-AnimeSharp.pth"]}]}}
        },
        "ImageUpscaleWithModel": {
            "input": {
                "required": {"upscale_model": ["UPSCALE_MODEL", {}], "image": ["IMAGE", {}]}
            }
        },
        "ETN_SaveImageCache": {"input": {"required": {"format": ["COMBO", {"options": ["PNG"]}]}}},
        "ImageScale": {
            "input": {
                "required": {
                    "image": ["IMAGE", {}],
                    "upscale_method": ["COMBO", {"options": ["lanczos"]}],
                    "width": ["INT", {"default": 1024}],
                    "height": ["INT", {"default": 1024}],
                    "crop": ["COMBO", {"options": ["disabled"]}],
                }
            }
        },
    }
    return ComfyObjectInfo(nodes)


def client_models() -> ClientModels:
    models = ClientModels()
    models.node_inputs = node_defs()
    models.checkpoints = {
        CHECKPOINT: CheckpointInfo(CHECKPOINT, resources.Arch.h3, FileFormat.diffusion)
    }
    models.resources = {
        resources.resource_id(
            resources.ResourceKind.text_encoder, resources.Arch.all, "minimax"
        ): TEXT_ENCODER,
        resources.resource_id(resources.ResourceKind.vae, resources.Arch.h3, "default"): VAE,
    }
    return models


def with_h3_image_studio(models: ClientModels) -> ClientModels:
    """Same client, but the server also has the H3 Image Studio image-to-image node."""
    defs = node_defs()
    defs.nodes["H3ImagePrepare"] = {
        "input": {
            "required": {
                "clip": ["CLIP", {}],
                "vae": ["VAE", {}],
                "source_image": ["IMAGE", {}],
                "prompt": ["STRING", {"default": ""}],
                "width": ["INT", {"default": 1024}],
                "height": ["INT", {"default": 1024}],
                "mode": [
                    "COMBO",
                    {"options": ["image_to_image (FL2VA)", "reference_edit (REF2VA)"]},
                ],
                "frame_preset": ["COMBO", {"options": ["single image | 1 frame (image VAE)"]}],
                "optimize_prompt": ["BOOLEAN", {"default": True}],
                "preserve_strength": ["FLOAT", {"default": 0.6}],
                "source_fit": ["COMBO", {"options": ["crop_center"]}],
                "reference_size": ["COMBO", {"options": ["match_generation_area"]}],
            },
            "optional": {
                **{f"reference_image_{i}": ["IMAGE", {}] for i in range(2, 10)},
            },
        },
        "output": ["CONDITIONING", "LATENT"],
    }
    models.node_inputs = defs
    return models


def sampling(sampler="er_sde", scheduler="manual", steps=8, cfg=1.0, start_step=0) -> SamplingInput:
    return SamplingInput(sampler, scheduler, cfg, steps, start_step, 1234)


def misc(batch=1):
    return workflow.MiscParams(batch, 1, 0.0, 0.0)


def prompt(positive="a lighthouse at dusk"):
    return workflow.Conditioning.from_input(ConditioningInput(positive, ""), sampling())


# --- detection --------------------------------------------------------------


def test_arch_detected_from_filename():
    assert resources.Arch.from_string("unknown", "eps", CHECKPOINT) is resources.Arch.h3
    assert resources.Arch.from_string("unknown", "eps", "P-H3-Image_v1.safetensors") is (
        resources.Arch.h3
    )


def test_h3_finetunes_are_the_h3_arch():
    names = (
        "DasiwaMinimaxH3_dasiwaHybridV4_int8.safetensors",
        "h3ErosMax_beta7_3346133.safetensors",
        "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    )
    for name in names:
        result = resources.Arch.from_string("unknown", "eps", name)
        assert result is resources.Arch.h3, name


def test_base_model_value_from_server():
    # comfyui-tooling-nodes reports "minimax-h3" once model_names knows the MiniMaxH3 class,
    # so the arch must resolve without relying on the filename.
    assert resources.Arch.from_string("minimax-h3", "eps", "whatever.safetensors") is (
        resources.Arch.h3
    )


def test_h3_arch_settings():
    for arch in (resources.Arch.h3,):
        assert arch.supports_edit
        assert not arch.is_edit
        assert not arch.supports_cfg
        assert arch.text_encoders == ["minimax"]
        assert arch in resources.Arch.list()

    from ai_diffusion.backend import resolution

    assert resolution.diffusion_multiple(resources.Arch.h3) == 32


def test_arch_flags():
    arch = resources.Arch.h3
    assert arch.supports_edit
    assert not arch.is_edit
    assert not arch.supports_cfg
    assert not arch.supports_regions
    assert not arch.has_controlnet_inpaint
    assert arch.text_encoders == ["minimax"]
    assert arch in resources.Arch.list()


def test_diffusion_multiple_is_32():
    from ai_diffusion.backend import resolution

    assert resolution.diffusion_multiple(resources.Arch.h3) == 32


# --- prompt shape -----------------------------------------------------------


def class_types(w: ComfyWorkflow):
    return [node["class_type"] for node in w.root.values()]


def nodes_of(w: ComfyWorkflow, class_type: str):
    return [n for n in w.root.values() if n["class_type"] == class_type]


def test_text_to_image_graph():
    models = client_models()
    w = ComfyWorkflow(node_defs=models.node_inputs)
    extent = ScaledExtent.no_scaling(Extent(1024, 768))
    w = workflow.generate_h3(
        w,
        CheckpointInput(CHECKPOINT, resources.Arch.h3),
        extent,
        prompt(),
        sampling(),
        misc(),
        models.for_arch(resources.Arch.h3),
    )

    types = class_types(w)
    assert "PH3ImageConditioning" in types
    assert "MiniMaxH3SigmaShift" in types
    assert "BasicGuider" in types
    assert "SamplerCustomAdvanced" in types
    assert "ManualSigmas" in types
    assert "KSampler" not in types
    assert "EmptyLatentImage" not in types
    assert "RepeatLatentBatch" not in types  # batch 1

    (ph3,) = nodes_of(w, "PH3ImageConditioning")
    assert ph3["inputs"]["width"] == 1024
    assert ph3["inputs"]["height"] == 768
    assert "images.image_1" not in ph3["inputs"]

    (sigmas,) = nodes_of(w, "ManualSigmas")
    values = [float(v) for v in sigmas["inputs"]["sigmas"].split(",")]
    assert len(values) == 9  # 8 sampling steps
    assert math.isclose(values[0], 1.0) and math.isclose(values[-1], 0.0)

    (sampler,) = nodes_of(w, "KSamplerSelect")
    assert sampler["inputs"]["sampler_name"] == "er_sde"
    assert "BasicScheduler" not in types  # agenda destilada, no scheduler de ComfyUI


def test_sampler_is_configurable():
    models = client_models()
    w = ComfyWorkflow(node_defs=models.node_inputs)
    workflow.generate_h3(
        w,
        CheckpointInput(CHECKPOINT, resources.Arch.h3),
        ScaledExtent.no_scaling(Extent(512, 512)),
        prompt(),
        sampling(sampler="dpmpp_2m"),
        misc(),
        models.for_arch(resources.Arch.h3),
    )
    (sampler,) = nodes_of(w, "KSamplerSelect")
    assert sampler["inputs"]["sampler_name"] == "dpmpp_2m"
    assert "ManualSigmas" in class_types(w)  # la agenda sigue siendo la destilada


def test_unknown_sampler_falls_back_to_distilled_default():
    models = client_models()
    w = ComfyWorkflow(node_defs=models.node_inputs)
    workflow.generate_h3(
        w,
        CheckpointInput(CHECKPOINT, resources.Arch.h3),
        ScaledExtent.no_scaling(Extent(512, 512)),
        prompt(),
        sampling(sampler="not_a_real_sampler"),
        misc(),
        models.for_arch(resources.Arch.h3),
    )
    (sampler,) = nodes_of(w, "KSamplerSelect")
    assert sampler["inputs"]["sampler_name"] == "er_sde"


def test_scheduler_and_steps_are_configurable():
    models = client_models()
    w = ComfyWorkflow(node_defs=models.node_inputs)
    workflow.generate_h3(
        w,
        CheckpointInput(CHECKPOINT, resources.Arch.h3),
        ScaledExtent.no_scaling(Extent(512, 512)),
        prompt(),
        sampling(sampler="dpmpp_2m", scheduler="karras", steps=12),
        misc(),
        models.for_arch(resources.Arch.h3),
    )
    types = class_types(w)
    assert "ManualSigmas" not in types
    (sched,) = nodes_of(w, "BasicScheduler")
    assert sched["inputs"]["scheduler"] == "karras"
    assert sched["inputs"]["steps"] == 12


def test_cfg_adds_a_second_conditioning_and_cfg_guider():
    models = client_models()
    w = ComfyWorkflow(node_defs=models.node_inputs)
    workflow.generate_h3(
        w,
        CheckpointInput(CHECKPOINT, resources.Arch.h3),
        ScaledExtent.no_scaling(Extent(512, 512)),
        prompt(),
        sampling(cfg=4.0),
        misc(),
        models.for_arch(resources.Arch.h3),
    )
    types = class_types(w)
    assert types.count("PH3ImageConditioning") == 2
    (guider,) = nodes_of(w, "CFGGuider")
    assert math.isclose(guider["inputs"]["cfg"], 4.0)
    assert "BasicGuider" not in types


def test_default_cfg_uses_basic_guider():
    models = client_models()
    w = ComfyWorkflow(node_defs=models.node_inputs)
    workflow.generate_h3(
        w,
        CheckpointInput(CHECKPOINT, resources.Arch.h3),
        ScaledExtent.no_scaling(Extent(512, 512)),
        prompt(),
        sampling(),
        misc(),
        models.for_arch(resources.Arch.h3),
    )
    types = class_types(w)
    assert types.count("PH3ImageConditioning") == 1
    assert "BasicGuider" in types
    assert "CFGGuider" not in types


def test_strength_does_not_change_the_graph():
    models = client_models()
    e = Extent(512, 512)

    def graph(start_step: int):
        wf_input = workflow.WorkflowInput(
            workflow.WorkflowKind.refine,
            ImageInput(ExtentInput(e, e, e, e), Image.create(e)),
            models=CheckpointInput(CHECKPOINT, resources.Arch.h3),
            conditioning=ConditioningInput("add snow", ""),
            sampling=sampling(start_step=start_step),
            batch_count=1,
        )
        return workflow.create(wf_input, models)

    full = graph(0)
    partial = graph(4)
    assert class_types(full) == class_types(partial)
    assert "ImageBlend" not in class_types(partial)
    sigmas = nodes_of(partial, "ManualSigmas")
    assert len(sigmas[0]["inputs"]["sigmas"].split(",")) == 9  # siempre la agenda destilada


def test_clip_loader_uses_minimax_type():
    models = client_models()
    w = ComfyWorkflow(node_defs=models.node_inputs)
    workflow.generate_h3(
        w,
        CheckpointInput(CHECKPOINT, resources.Arch.h3),
        ScaledExtent.no_scaling(Extent(512, 512)),
        prompt(),
        sampling(),
        misc(),
        models.for_arch(resources.Arch.h3),
    )
    (clip,) = nodes_of(w, "CLIPLoader")
    assert clip["inputs"]["clip_name"] == TEXT_ENCODER
    assert clip["inputs"]["type"] == "minimax"


def test_edit_graph_passes_reference_image():
    models = client_models()
    w = ComfyWorkflow(node_defs=models.node_inputs)
    w = workflow.refine_h3(
        w,
        Image.create(Extent(64, 64)),
        ScaledExtent.no_scaling(Extent(64, 64)),
        CheckpointInput(CHECKPOINT, resources.Arch.h3),
        prompt("Edit <Picture 1>. Change the sky to sunset."),
        sampling(),
        misc(),
        models.for_arch(resources.Arch.h3),
    )
    (ph3,) = nodes_of(w, "PH3ImageConditioning")
    assert "images.image_1" in ph3["inputs"]
    assert isinstance(ph3["inputs"]["images.image_1"], list)


def test_batch_uses_repeat_latent_batch():
    models = client_models()
    w = ComfyWorkflow(node_defs=models.node_inputs)
    workflow.generate_h3(
        w,
        CheckpointInput(CHECKPOINT, resources.Arch.h3),
        ScaledExtent.no_scaling(Extent(512, 512)),
        prompt(),
        sampling(),
        misc(batch=3),
        models.for_arch(resources.Arch.h3),
    )
    assert "RepeatLatentBatch" in class_types(w)


def test_selection_crops_and_composites():
    models = client_models()
    e = Extent(512, 512)
    wf_input = workflow.WorkflowInput(
        workflow.WorkflowKind.inpaint,
        ImageInput(ExtentInput(e, e, e, e), Image.create(e), Image.create(e)),
        models=CheckpointInput(CHECKPOINT, resources.Arch.h3),
        sampling=sampling(),
        conditioning=ConditioningInput("make it snowy", ""),
        inpaint=InpaintParams(InpaintMode.custom, Bounds(128, 128, 256, 256), grow=8, feather=8),
        crop_upscale_extent=e,
    )
    assert wf_input.images is not None
    wf_input.images.hires_mask = mask_with_rect(e, Bounds(128, 128, 256, 256))
    w = workflow.create(wf_input, models)

    types = class_types(w)
    assert "ImageCrop" in types  # only the selection goes to the model
    assert "CropMask" in types
    assert "ETN_ApplyMaskToImage" in types  # result pasted back through the mask
    assert "KSampler" not in types

    (ph3,) = nodes_of(w, "PH3ImageConditioning")
    assert "images.image_1" in ph3["inputs"]
    # 256 selection + 64 px of context on each side: the model needs to see the surroundings.
    # Then rendered at 4 MP (capped at 4x linear) - 384 * 4 = 1536.
    assert ph3["inputs"]["width"] == 1536
    assert ph3["inputs"]["height"] == 1536
    assert class_types(w).count("ImageCrop") == 2  # context crop in, selection crop back out
    # The patch goes back into the untouched image at the selection offset, so nothing else moves.
    (composite,) = nodes_of(w, "ImageCompositeMasked")
    assert composite["inputs"]["x"] == 128 and composite["inputs"]["y"] == 128

    # Compositing uses the mask the user drew, not the grown/feathered denoise mask, otherwise the
    # invented pixels land outside the selection.
    (apply,) = nodes_of(w, "ETN_ApplyMaskToImage")
    mask_node_id = apply["inputs"]["mask"][0]
    chain = [w.root[mask_node_id]["class_type"]]
    while node_inputs := [
        v for v in w.root[mask_node_id]["inputs"].values() if isinstance(v, list)
    ]:
        mask_node_id = node_inputs[0][0]
        chain.append(w.root[mask_node_id]["class_type"])
    assert "INPAINT_ExpandMask" not in chain
    assert chain[0] == "ThresholdMask" and chain[1] == "CropMask"


def test_region_layers_are_rejected():
    from ai_diffusion.backend.api import RegionInput
    from ai_diffusion.image import Bounds

    cond = workflow.Conditioning.from_input(
        ConditioningInput(
            "a cat",
            "",
            regions=[
                RegionInput(
                    mask=Image.create(Extent(256, 256)),
                    bounds=Bounds(0, 0, 256, 256),
                    positive="a dog",
                )
            ],
        ),
        sampling(),
    )
    with pytest.raises(PluginError, match="region"):
        workflow._h3_no_regions(cond)


def test_refine_with_a_selection_is_treated_as_a_crop_edit():
    models = client_models()
    w = ComfyWorkflow(node_defs=models.node_inputs)
    e = Extent(512, 512)
    wf_input = workflow.WorkflowInput(
        workflow.WorkflowKind.refine_region,
        ImageInput(ExtentInput(e, e, e, e), Image.create(e)),
        models=CheckpointInput(CHECKPOINT, resources.Arch.h3),
        conditioning=ConditioningInput("make it snow", ""),
        sampling=sampling(),
        inpaint=InpaintParams(InpaintMode.fill, Bounds(128, 128, 256, 256)),
        batch_count=1,
    )
    assert wf_input.images is not None
    wf_input.images.hires_mask = Image.create(e)
    w = workflow.create(wf_input, models)
    types = class_types(w)
    assert "ImageCrop" in types
    assert "PH3ImageConditioning" in types
    assert "ETN_ApplyMaskToImage" in types


def test_builtin_style_matches_the_h3_family():
    from ai_diffusion.backend.client import resolve_arch
    from ai_diffusion.style import Style, Styles

    path = Styles.default_builtin_folder / "minimax-h3.json"
    assert path.exists()
    style = Style.load(path)
    assert style is not None
    assert style.sampler == h3_sampler_preset
    assert style.sampler_steps == 8
    assert math.isclose(style.cfg_scale, 1.0)

    models = client_models()
    assert style.preferred_checkpoint(models.checkpoints.keys()) == CHECKPOINT
    assert resolve_arch(style, models) is resources.Arch.h3


def test_prompt_json_is_serializable():
    models = client_models()
    w = ComfyWorkflow(node_defs=models.node_inputs)
    workflow.generate_h3(
        w,
        CheckpointInput(CHECKPOINT, resources.Arch.h3),
        ScaledExtent.no_scaling(Extent(512, 512)),
        prompt(),
        sampling(),
        misc(),
        models.for_arch(resources.Arch.h3),
    )
    json.dumps(w.root)  # must not raise


def test_edit_seeds_the_latent_from_the_image_when_available():
    models = with_h3_image_studio(client_models())
    e = Extent(512, 512)
    wf_input = workflow.WorkflowInput(
        workflow.WorkflowKind.refine,
        ImageInput(ExtentInput(e, e, e, e), Image.create(e)),
        models=CheckpointInput(CHECKPOINT, resources.Arch.h3),
        conditioning=ConditioningInput("add snow", ""),
        sampling=sampling(),
        batch_count=1,
    )
    w = workflow.create(wf_input, models)

    (prepare,) = nodes_of(w, "H3ImagePrepare")
    assert prepare["inputs"]["mode"] == "image_to_image (FL2VA)"
    assert prepare["inputs"]["frame_preset"] == "single image | 1 frame (image VAE)"
    assert math.isclose(prepare["inputs"]["preserve_strength"], 0.6)
    assert "PH3ImageConditioning" not in class_types(w)
    # El latent viene del nodo, no de un VAEEncode: no se puede codificar aparte.
    assert "VAEEncode" not in class_types(w)


# --- references (REF2VA) ----------------------------------------------------

def refs(count: int) -> list:
    return [
        ControlInput(resources.ControlMode.reference, Image.create(Extent(32, 32)), 1.0)
        for _ in range(count)
    ]


def edit_workflow(models, reference_count: int):
    e = Extent(512, 512)
    wf_input = workflow.WorkflowInput(
        workflow.WorkflowKind.refine,
        ImageInput(ExtentInput(e, e, e, e), Image.create(e)),
        models=CheckpointInput(CHECKPOINT, resources.Arch.h3),
        conditioning=ConditioningInput("make the dress red", "", control=refs(reference_count)),
        sampling=sampling(),
        batch_count=1,
    )
    return workflow.create(wf_input, models)


def generate_workflow(models, reference_count: int):
    e = Extent(512, 512)
    wf_input = workflow.WorkflowInput(
        workflow.WorkflowKind.generate,
        ImageInput(ExtentInput(e, e, e, e), Image.create(e)),
        models=CheckpointInput(CHECKPOINT, resources.Arch.h3),
        conditioning=ConditioningInput("a woman in a red dress", "", control=refs(reference_count)),
        sampling=sampling(),
        batch_count=1,
    )
    return workflow.create(wf_input, models)


def test_edit_with_references_switches_to_ref2va():
    models = with_h3_image_studio(client_models())
    w = edit_workflow(models, 2)

    (prepare,) = nodes_of(w, "H3ImagePrepare")
    inputs = prepare["inputs"]
    assert inputs["mode"] == "reference_edit (REF2VA)"
    assert "reference_image_2" in inputs
    assert "reference_image_3" in inputs
    assert "reference_image_4" not in inputs
    # El lienzo sigue siendo <Picture 1>.
    assert isinstance(inputs["source_image"], list)
    assert "ETN_LoadImageCache" in class_types(w)


def test_edit_reference_prompt_keeps_picture_1_in_charge():
    models = with_h3_image_studio(client_models())
    w = edit_workflow(models, 1)

    (prepare,) = nodes_of(w, "H3ImagePrepare")
    prompt = prepare["inputs"]["prompt"]
    assert prompt.startswith("Edit <Picture 1>, the picture currently on the canvas.")
    assert "The instruction refers to <Picture 1> unless it names another picture." in prompt
    assert "Apply exactly the changes it asks for and leave the rest of <Picture 1> unchanged" in prompt
    assert prompt.endswith("make the dress red")


def test_edit_without_references_stays_image_to_image():
    models = with_h3_image_studio(client_models())
    w = edit_workflow(models, 0)

    (prepare,) = nodes_of(w, "H3ImagePrepare")
    assert prepare["inputs"]["mode"] == "image_to_image (FL2VA)"
    assert not any(k.startswith("reference_image_") for k in prepare["inputs"])
    assert not prepare["inputs"]["prompt"].startswith("Edit <Picture 1>.")


def test_generate_with_references_builds_the_subject_from_them():
    models = with_h3_image_studio(client_models())
    w = generate_workflow(models, 2)

    (prepare,) = nodes_of(w, "H3ImagePrepare")
    inputs = prepare["inputs"]
    assert inputs["mode"] == "reference_edit (REF2VA)"
    assert "reference_image_2" in inputs
    assert "reference_image_3" not in inputs  # la primera referencia es <Picture 1>
    assert inputs["prompt"].startswith("Create one finished still image")
    assert "<Picture 1>, <Picture 2>" in inputs["prompt"]


def test_more_than_eight_references_is_rejected():
    models = with_h3_image_studio(client_models())
    with pytest.raises(PluginError, match="at most 8 reference layers"):
        edit_workflow(models, 9)


def test_references_reach_the_fallback_node_too():
    # Sin el pack de H3 Image Studio, el nodo p-h3 recibe las mismas imágenes como autogrow.
    models = client_models()
    w = edit_workflow(models, 2)

    (ph3,) = nodes_of(w, "PH3ImageConditioning")
    assert "images.image_1" in ph3["inputs"]
    assert "images.image_2" in ph3["inputs"]
    assert "images.image_3" in ph3["inputs"]


# --- <layer:name> syntax ----------------------------------------------------

def test_layer_placeholders_become_picture_tags():
    from ai_diffusion.text import replace_layers

    template = workflow._layer_replace[resources.Arch.h3]
    text = "Change her hair to the colour of <layer:Hair> and the dress from <layer:Dress>."
    assert replace_layers(text, {"Hair": 2, "Dress": 3}, template) == (
        "Change her hair to the colour of <Picture 2> and the dress from <Picture 3>."
    )


def test_layer_placeholders_are_numbered_after_the_canvas():
    from ai_diffusion.text import extract_layers

    text = "<layer:Hair> plus <layer:Dress> plus <layer:Hair> again"
    editing = ConditioningInput(text, "", edit_reference=True)
    generating = ConditioningInput(text, "", edit_reference=False)
    assert extract_layers(editing) == {"Hair": 2, "Dress": 3}
    assert extract_layers(generating) == {"Hair": 1, "Dress": 2}


def test_control_net_layers_are_not_silently_rewritten_for_h3():
    """H3 has no ControlNet: pose/depth layers stay unsupported instead of becoming references."""
    e = Extent(64, 64)
    cond = ConditioningInput(
        "keep the pose",
        "",
        control=[ControlInput(resources.ControlMode.pose, Image.create(e), 1.0)],
        edit_reference=True,
    )
    assert not resources.ControlMode.pose.can_substitute_instruction(resources.Arch.h3)
    assert workflow.build_instructions(cond, resources.Arch.h3, None) == "keep the pose"
    assert cond.control[0].mode is resources.ControlMode.pose


def test_prompt_with_layer_names_reaches_the_conditioning_as_picture_tags():
    from ai_diffusion.style import Style

    cond = ConditioningInput(
        "Change only the hair to the colour of <layer:Hair>, keep <Picture 1> intact.",
        "",
        control=[ControlInput(resources.ControlMode.reference, Image.create(Extent(32, 32)), 1.0)],
        edit_reference=True,
    )
    prepared, _, meta = workflow.prepare_prompts(
        cond, Style(Path("h3-test.json")), 1234, resources.Arch.h3, None, {"Hair": 2}
    )
    assert "<Picture 2>" in prepared.positive
    assert "<layer:" not in prepared.positive
    assert "<Picture 2>" in meta["prompt_final"]


def test_edit_prompt_does_not_have_to_name_the_canvas():
    """The guide states which picture the instruction refers to, so <Picture 1> is optional."""
    models = with_h3_image_studio(client_models())
    w = edit_workflow(models, 1)

    (prepare,) = nodes_of(w, "H3ImagePrepare")
    prompt = prepare["inputs"]["prompt"]
    body = prompt.split("\n\n", 1)[1]
    assert "<Picture 1>" not in body  # el usuario no lo escribió
    assert prompt.count("<Picture 1>") >= 2  # el guía lo dice dos veces


def test_prompt_upsampler_runs_as_its_own_job(monkeypatch):
    """The upsampler and the model cannot share VRAM, so it is chained as a separate prompt."""
    from ai_diffusion.backend import comfy_client

    client = comfy_client.ComfyClient("http://127.0.0.1:8188")
    models = client_models()
    defs = node_defs()
    defs.nodes["TextGenerate"] = {
        "input": {
            "required": {
                "clip": ["CLIP", {}],
                "prompt": ["STRING", {"default": ""}],
                "max_length": ["INT", {"default": 512}],
                "sampling_mode": ["COMBO", {"options": ["off"]}],
            }
        },
        "output": ["STRING", "STRING"],
    }
    models.node_inputs = defs
    models.resources = {
        **models.resources,
        "upsampler": "p-h3-image_prompt_upsampler_int8_convrot.safetensors",
    }
    client.models = models

    sent = {}

    async def fake_post(op, data):
        sent[op] = data
        return {"prompt_id": data["prompt_id"]}

    async def fake_get(op, timeout=60):
        if op.startswith("history/"):
            prompt_id = op.split("/", 1)[1]
            node_ids = [
                k for k, v in sent["prompt"]["prompt"].items() if v["class_type"] == "PreviewAny"
            ]
            return {
                prompt_id: {
                    "status": {"status_str": "success"},
                    "outputs": {node_ids[0]: {"text": ["EXPANDED BRIEF"]}},
                }
            }
        return {}

    monkeypatch.setattr(client, "_post", fake_post)
    monkeypatch.setattr(client, "_get", fake_get)

    e = Extent(512, 512)
    work = workflow.WorkflowInput(
        workflow.WorkflowKind.refine,
        ImageInput(ExtentInput(e, e, e, e), Image.create(e)),
        models=CheckpointInput(CHECKPOINT, resources.Arch.h3),
        conditioning=ConditioningInput("make her smile", ""),
        sampling=sampling(),
        batch_count=1,
    )
    work.expand_prompt = True
    result = asyncio.run(client.expand_h3_prompt(work))

    assert result.conditioning is not None
    assert result.conditioning.positive == "EXPANDED BRIEF"
    graph = sent["prompt"]["prompt"]
    (generate,) = [n for n in graph.values() if n["class_type"] == "TextGenerate"]
    assert "<Picture 1>" in generate["inputs"]["prompt"]
    assert "make her smile" in generate["inputs"]["prompt"]
    assert generate["inputs"]["sampling_mode"] == "off"
    # El job principal sigue recibiendo el texto ya expandido, sin el nodo de texto.
    assert all(
        n["class_type"] != "TextGenerate"
        for n in workflow.create(
            dataclasses.replace(result, expand_prompt=False), models
        ).root.values()
    )


def test_upscale_without_refine_needs_no_checkpoint():
    """Unchecking "Refine upscaled image" builds a graph with no checkpoint at all.

    The H3 branch reads the architecture from the checkpoint, which an upscale job does not carry,
    so it used to fail with the useless message "a value is required".
    """
    models = client_models()
    e = Extent(256, 256)
    wf_input = workflow.prepare_upscale_simple(Image.create(e), "4x-AnimeSharp.pth", 2.0)
    w = workflow.create(wf_input, models)

    (loader,) = nodes_of(w, "UpscaleModelLoader")
    assert loader["inputs"]["model_name"] == "4x-AnimeSharp.pth"
    assert "ImageUpscaleWithModel" in class_types(w)
    assert "KSampler" not in class_types(w)


def test_filename_only_decides_when_the_server_says_unknown():
    """Same rule as Anima: a filename must not override what the server reported."""
    assert (
        resources.Arch.from_string("sdxl", "eps", "my_h3_style.safetensors") is resources.Arch.sdxl
    )
    assert resources.Arch.from_string("flux", "eps", "minimax_lora.safetensors") is (
        resources.Arch.flux
    )
