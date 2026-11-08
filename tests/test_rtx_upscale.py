"""RTX Video Super Resolution upscaling via the official NVIDIA VFX node pack.

The node schema changed between pack versions (quality modes were expanded, ``strength`` and
``image_encoding`` were added), so behavior is derived from ``object_info`` at runtime instead of
being hardcoded. These tests pin that contract without requiring a running ComfyUI server.
"""

import math

from ai_diffusion.backend import resources, workflow
from ai_diffusion.backend.api import UpscaleInput
from ai_diffusion.backend.client import ClientModels
from ai_diffusion.backend.comfy_client import _find_rtx_vsr
from ai_diffusion.backend.comfy_workflow import ComfyObjectInfo, ComfyWorkflow
from ai_diffusion.image import Extent, Image

quality_01x = ["LOW", "MEDIUM", "HIGH", "ULTRA"]
quality_02x = [
    "BICUBIC",
    "LOW",
    "MEDIUM",
    "HIGH",
    "ULTRA",
    "DENOISE_LOW",
    "DENOISE_MEDIUM",
    "DENOISE_HIGH",
    "DENOISE_ULTRA",
    "DEBLUR_LOW",
    "DEBLUR_MEDIUM",
    "DEBLUR_HIGH",
    "DEBLUR_ULTRA",
    "HIGHBITRATE_LOW",
    "HIGHBITRATE_MEDIUM",
    "HIGHBITRATE_HIGH",
    "HIGHBITRATE_ULTRA",
    "STREAMING_MEDIUM",
    "STREAMING_ULTRA",
]


def rtx_node_info(quality: list[str], strength=False, image_encoding=False):
    inputs: dict = {
        "images": ["IMAGE", {}],
        "resize_type": [
            "COMFY_DYNAMICCOMBO_V3",
            {
                "options": [
                    {
                        "key": "scale by multiplier",
                        "inputs": {
                            "required": {
                                "scale": ["FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}]
                            }
                        },
                    },
                    {
                        "key": "target dimensions",
                        "inputs": {
                            "required": {
                                "width": ["INT", {"default": 1920}],
                                "height": ["INT", {"default": 1080}],
                            }
                        },
                    },
                ]
            },
        ],
        "quality": ["COMBO", {"options": quality, "default": "ULTRA"}],
    }
    if strength:
        inputs["strength"] = ["FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}]
    if image_encoding:
        inputs["image_encoding"] = [
            "COMBO",
            {"options": ["8-bit RGB", "10-bit RGB (RGB10A2)"], "default": "8-bit RGB"},
        ]
    return {"input": {"required": inputs}, "output": ["IMAGE"]}


def object_info(rtx: dict | None, extra: dict | None = None) -> ComfyObjectInfo:
    nodes = {
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
        "UpscaleModelLoader": {"input": {"required": {"model_name": ["COMBO", {}]}}},
        "ImageUpscaleWithModel": {"input": {"required": {}}},
    }
    if rtx is not None:
        nodes[resources.rtx_vsr_node] = rtx
    if extra:
        nodes.update(extra)
    return ComfyObjectInfo(nodes)


def class_types(w: ComfyWorkflow):
    return [node["class_type"] for node in w.root.values()]


def rtx_nodes(w: ComfyWorkflow):
    return [n for n in w.root.values() if n["class_type"] == resources.rtx_vsr_node]


# --- detection / schema -----------------------------------------------------


def test_rtx_node_absent():
    models = ClientModels()
    models.upscalers = ["4x.pth"]
    info = _find_rtx_vsr(object_info(None))
    assert info is None
    assert resources.rtx_vsr_node not in models.upscalers


def test_rtx_node_legacy_schema_has_no_strength():
    info = _find_rtx_vsr(object_info(rtx_node_info(quality_01x)))
    assert info is not None
    assert info["quality"] == quality_01x
    assert info["default_quality"] == "ULTRA"
    assert info["has_strength"] is False


def test_rtx_node_current_schema():
    node = rtx_node_info(quality_02x, strength=True, image_encoding=True)
    info = _find_rtx_vsr(object_info(node))
    assert info is not None
    assert info["quality"] == quality_02x
    assert info["has_strength"] is True


def test_rtx_node_legacy_combo_list_form():
    # older ComfyUI object_info exposed combo values as the first element
    node = {"input": {"required": {"quality": [["LOW", "ULTRA"], {}]}}}
    info = _find_rtx_vsr(object_info(node))
    assert info is not None
    assert info["quality"] == ["LOW", "ULTRA"]
    assert info["default_quality"] == "LOW"


def test_rtx_models_roundtrip():
    models = ClientModels()
    models.rtx_vsr = {"quality": quality_01x, "default_quality": "ULTRA", "has_strength": True}
    restored = ClientModels.from_dict({"upscalers": ["4x.pth"], "rtx_vsr": models.rtx_vsr})
    assert restored.rtx_vsr == models.rtx_vsr
    assert restored.upscalers == ["4x.pth"]


# --- prompt generation ------------------------------------------------------


def test_rtx_prompt_inputs():
    defs = object_info(rtx_node_info(quality_01x))
    w = ComfyWorkflow(node_defs=defs)
    model = Image.create(Extent(64, 64))
    input = UpscaleInput(resources.rtx_vsr_node, rtx_quality="HIGH", rtx_strength=0.5)
    w = workflow.upscale_simple(w, model, input, 2.0)

    nodes = rtx_nodes(w)
    assert len(nodes) == 1
    inputs = nodes[0]["inputs"]
    assert inputs["resize_type"] == "scale by multiplier"
    assert math.isclose(inputs["resize_type.scale"], 2.0)
    assert inputs["quality"] == "HIGH"
    # not declared by this node version -> must not be sent
    assert "strength" not in inputs


def test_rtx_prompt_sends_strength_when_supported():
    node = rtx_node_info(quality_02x, strength=True, image_encoding=True)
    w = ComfyWorkflow(node_defs=object_info(node))
    model = Image.create(Extent(64, 64))
    input = UpscaleInput(resources.rtx_vsr_node, rtx_quality="DENOISE_LOW", rtx_strength=0.25)
    w = workflow.upscale_simple(w, model, input, 2.0)

    inputs = rtx_nodes(w)[0]["inputs"]
    assert inputs["quality"] == "DENOISE_LOW"
    assert math.isclose(inputs["strength"], 0.25)


def test_rtx_scale_factor_is_capped():
    w = ComfyWorkflow(node_defs=object_info(rtx_node_info(quality_01x)))
    model = Image.create(Extent(64, 64))
    input = UpscaleInput(resources.rtx_vsr_node)
    w = workflow.upscale_simple(w, model, input, 6.0)

    assert math.isclose(rtx_nodes(w)[0]["inputs"]["resize_type.scale"], 4.0)
    assert "ImageScale" in class_types(w)


def test_rtx_no_resize_needed_for_multiple_of_8():
    w = ComfyWorkflow(node_defs=object_info(rtx_node_info(quality_01x)))
    model = Image.create(Extent(64, 64))
    input = UpscaleInput(resources.rtx_vsr_node)
    w = workflow.upscale_simple(w, model, input, 2.0)

    assert "ImageScale" not in class_types(w)


def test_rtx_resize_to_non_multiple_of_8():
    w = ComfyWorkflow(node_defs=object_info(rtx_node_info(quality_01x)))
    model = Image.create(Extent(10, 10))
    input = UpscaleInput(resources.rtx_vsr_node)
    w = workflow.upscale_simple(w, model, input, 2.0)

    scale_nodes = [n for n in w.root.values() if n["class_type"] == "ImageScale"]
    assert len(scale_nodes) == 1
    assert scale_nodes[0]["inputs"]["width"] == 20
    assert scale_nodes[0]["inputs"]["height"] == 20


def test_rtx_denoise_mode_resizes_to_target():
    w = ComfyWorkflow(node_defs=object_info(rtx_node_info(quality_02x, strength=True)))
    model = Image.create(Extent(64, 64))
    input = UpscaleInput(resources.rtx_vsr_node, rtx_quality="DEBLUR_HIGH")
    w = workflow.upscale_simple(w, model, input, 2.0)

    # DENOISE/DEBLUR keep the input resolution, so the target extent is reached by resampling
    scale_nodes = [n for n in w.root.values() if n["class_type"] == "ImageScale"]
    assert len(scale_nodes) == 1
    assert scale_nodes[0]["inputs"]["width"] == 128


def test_regular_upscaler_unchanged():
    w = ComfyWorkflow(node_defs=object_info(None))
    model = Image.create(Extent(64, 64))
    input = UpscaleInput("4x_NMKD-Superscale-SP_178000_G.pth")
    w = workflow.upscale_simple(w, model, input, 2.0)

    types = class_types(w)
    assert "UpscaleModelLoader" in types
    assert "ImageUpscaleWithModel" in types
    assert resources.rtx_vsr_node not in types
    assert "ImageScale" in types  # 4x model then resample to the requested 2x
