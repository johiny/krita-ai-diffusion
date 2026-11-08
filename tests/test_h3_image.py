"""MiniMax H3 support: architecture detection and prompt shape.

The family (P-H3 Image distillation and its finetunes) uses one conditioning node that builds
both the conditioning and the latent, a fixed 8-step schedule and no CFG.
"""

import json
import math

import pytest

from ai_diffusion.backend import resources, workflow
from ai_diffusion.backend.api import (
    CheckpointInput,
    ConditioningInput,
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
        ImageInput(ExtentInput(e, e, e, e), Image.create(e), Image.create(e), Image.create(e)),
        models=CheckpointInput(CHECKPOINT, resources.Arch.h3),
        sampling=sampling(),
        conditioning=ConditioningInput("make it snowy", ""),
        inpaint=InpaintParams(InpaintMode.custom, Bounds(128, 128, 256, 256), grow=8, feather=8),
        crop_upscale_extent=e,
    )
    w = workflow.create(wf_input, models)

    types = class_types(w)
    assert "ImageCrop" in types  # only the selection goes to the model
    assert "CropMask" in types
    assert "ETN_ApplyMaskToImage" in types  # result pasted back through the mask
    assert "KSampler" not in types

    (ph3,) = nodes_of(w, "PH3ImageConditioning")
    assert "images.image_1" in ph3["inputs"]
    assert ph3["inputs"]["width"] == 256
    assert ph3["inputs"]["height"] == 256


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
