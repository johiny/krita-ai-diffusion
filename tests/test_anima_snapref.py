"""Anima reference layers are routed through the SnapRef node.

SnapRef keeps the reference K/V of every attention block for the sampler, so a reference
control layer on an Anima checkpoint has to wrap the model instead of the conditioning.
"""

import json

import pytest

from ai_diffusion.backend import resources, workflow
from ai_diffusion.backend.api import ControlInput
from ai_diffusion.backend.client import CheckpointInfo, ClientModels, ModelDict
from ai_diffusion.backend.comfy_workflow import ComfyObjectInfo, ComfyWorkflow
from ai_diffusion.backend.resources import Arch, ControlMode, ResourceKind
from ai_diffusion.image import Extent, Image

ANIMA = "anima_base_v1.safetensors"
ADAPTER = "snapref_v2a_fp8.safetensors"
STYLE_ADAPTER = "snapref_v2s_pre1_fp8.safetensors"


def node_defs() -> ComfyObjectInfo:
    return ComfyObjectInfo(
        {
            "CheckpointLoaderSimple": {
                "input": {"required": {"ckpt_name": ["COMBO", {}]}},
                "output": ["MODEL", "CLIP", "VAE"],
            },
            "ETN_LoadImageCache": {
                "input": {"required": {"id": ["STRING", {}]}},
                "output": ["IMAGE", "MASK"],
            },
            "SnapRef": {
                "input": {"required": {"model": ["MODEL", {}], "image": ["IMAGE", {}]}},
                "output": ["MODEL"],
            },
        }
    )


def make_models(
    loras: list[str],
    arch: Arch = Arch.anima,
    adapters: dict[str, list[str]] | None = None,
) -> ModelDict:
    client_models = ClientModels()
    client_models.loras = list(loras)
    client_models.snapref_adapters = dict(adapters or {})
    client_models.node_inputs = node_defs()
    return ModelDict(client_models, ResourceKind.checkpoint, arch)


def reference_layers(count: int, strength: float = 1.0) -> list[workflow.Control]:
    return [
        workflow.Control.from_input(
            ControlInput(ControlMode.reference, Image.create(Extent(64, 64)), strength)
        )
        for _ in range(count)
    ]


def build(
    loras: list[str],
    count: int,
    arch: Arch = Arch.anima,
    strength: float = 1.0,
    adapters: dict[str, list[str]] | None = None,
):
    models = make_models(loras, arch, adapters)
    w = ComfyWorkflow(node_defs=models.node_inputs)
    model, clip, vae = w.add("CheckpointLoaderSimple", 3, ckpt_name=ANIMA)
    # The real builder hands over the Clip wrapper, which carries the arch with it.
    model = workflow.apply_anima_references(
        w, model, reference_layers(count, strength), workflow.Clip(clip, arch), vae, models
    )
    assert_serializable(w)
    return w, model


def assert_serializable(w: ComfyWorkflow):
    """The prompt is posted as JSON, so no node input may hold a non-serializable object."""
    json.dumps(w.root)


def nodes_of(w: ComfyWorkflow, class_type: str):
    return [(key, node) for key, node in w.root.items() if node["class_type"] == class_type]


def test_reference_layer_wraps_the_model_with_snapref():
    w, model = build([ADAPTER], count=1)

    (key, node), *rest = nodes_of(w, "SnapRef")
    assert not rest
    assert node["inputs"]["adapter"] == ADAPTER
    assert node["inputs"]["weight_1"] == pytest.approx(1.0)
    assert node["inputs"]["mode"] == "cs"
    (_, _checkpoint) = nodes_of(w, "CheckpointLoaderSimple")[0]
    assert node["inputs"]["clip"] == [checkpoint_key(w, "CheckpointLoaderSimple"), 1]
    assert str(model.node) == key, "the sampler must read the SnapRef output"


def checkpoint_key(w: ComfyWorkflow, class_type: str) -> str:
    return nodes_of(w, class_type)[0][0]


def test_reference_image_comes_from_the_control_layer():
    w, _ = build([ADAPTER], count=1)

    (_, snapref), *rest = nodes_of(w, "SnapRef")
    (_, _load_image), *other = nodes_of(w, "ETN_LoadImageCache")
    assert not rest and not other
    assert snapref["inputs"]["image"] == [checkpoint_key(w, "ETN_LoadImageCache"), 0]


def test_layer_strength_maps_to_the_linear_weight():
    w, _ = build([ADAPTER], count=1, strength=0.5)

    (_, snapref) = nodes_of(w, "SnapRef")[0]
    assert snapref["inputs"]["weight_1"] == pytest.approx(0.5)


def test_several_reference_layers_share_one_node():
    w, model = build([ADAPTER], count=2)

    (key, node), *rest = nodes_of(w, "SnapRef")
    assert not rest, "one SnapRef node reads every reference of a generation"
    assert str(model.node) == key
    images = nodes_of(w, "ETN_LoadImageCache")
    assert len(images) == 2
    assert node["inputs"]["image"] == [images[0][0], 0]
    assert node["inputs"]["image_2"] == [images[1][0], 0]
    assert "image_3" not in node["inputs"]


def test_each_reference_keeps_its_own_weight():
    models = make_models([ADAPTER])
    w = ComfyWorkflow(node_defs=models.node_inputs)
    model, clip, vae = w.add("CheckpointLoaderSimple", 3, ckpt_name=ANIMA)
    layers = [
        workflow.Control.from_input(
            ControlInput(ControlMode.reference, Image.create(Extent(64, 64)), weight)
        )
        for weight in (1.0, 0.4)
    ]

    model = workflow.apply_anima_references(
        w, model, layers, workflow.Clip(clip, Arch.anima), vae, models
    )
    assert_serializable(w)

    (_, node) = nodes_of(w, "SnapRef")[0]
    assert node["inputs"]["weight_1"] == pytest.approx(1.0)
    assert node["inputs"]["weight_2"] == pytest.approx(0.4)


def _reference_layers(weights):
    models = make_models([ADAPTER])
    w = ComfyWorkflow(node_defs=models.node_inputs)
    model, clip, vae = w.add("CheckpointLoaderSimple", 3, ckpt_name=ANIMA)
    layers = [
        workflow.Control.from_input(
            ControlInput(ControlMode.reference, Image.create(Extent(64, 64)), weight)
        )
        for weight in weights
    ]
    return w, models, model, clip, vae, layers


def test_a_reference_with_zero_strength_is_left_out():
    w, models, model, clip, vae, layers = _reference_layers((1.0, 0.0))

    model = workflow.apply_anima_references(
        w, model, layers, workflow.Clip(clip, Arch.anima), vae, models
    )
    assert_serializable(w)

    (_, node) = nodes_of(w, "SnapRef")[0]
    assert "image_2" not in node["inputs"], "a zero weight would make the log bias -inf"
    assert len(nodes_of(w, "ETN_LoadImageCache")) == 1


def test_no_reference_node_when_every_layer_is_muted():
    w, models, model, clip, vae, layers = _reference_layers((0.0, 0.0))

    result = workflow.apply_anima_references(
        w, model, layers, workflow.Clip(clip, Arch.anima), vae, models
    )

    assert not nodes_of(w, "SnapRef")
    assert str(result.node) == str(model.node)


def test_three_references_fill_every_slot():
    w, _ = build([ADAPTER], count=3)

    (_, node) = nodes_of(w, "SnapRef")[0]
    assert "image_3" in node["inputs"]
    assert node["inputs"]["weight_3"] == pytest.approx(1.0)
    assert not nodes_of(w, "ImageStitch")


def test_four_references_are_packed_into_one_picture():
    w, _ = build([ADAPTER], count=4)

    (_, node) = nodes_of(w, "SnapRef")[0]
    stitches = nodes_of(w, "ImageStitch")
    assert len(stitches) == 3, "each extra picture joins the packed reference"
    assert node["inputs"]["image"] == [stitches[-1][0], 0]
    assert "image_2" not in node["inputs"]
    assert "image_3" not in node["inputs"]
    assert node["inputs"]["weight_1"] == pytest.approx(1.0)
    for key, stitch in stitches:
        assert stitch["inputs"]["match_image_size"] is True
        for name in ("image1", "image2"):
            link = stitch["inputs"][name]
            assert isinstance(link, list) and len(link) == 2, (key, name, link)
            assert link[0] in w.root, (key, name, link)


def test_the_default_adapter_is_the_one_that_reads_style():
    modes = {ADAPTER: ["cs", "char"], STYLE_ADAPTER: ["cs", "char", "style"]}
    w, _ = build([ADAPTER, STYLE_ADAPTER], count=1, adapters=modes)

    (_, snapref) = nodes_of(w, "SnapRef")[0]
    assert snapref["inputs"]["adapter"] == STYLE_ADAPTER


def test_without_adapter_metadata_the_latest_name_is_used():
    w, _ = build([ADAPTER, STYLE_ADAPTER], count=1)

    (_, snapref) = nodes_of(w, "SnapRef")[0]
    assert snapref["inputs"]["adapter"] == STYLE_ADAPTER


def test_the_layer_can_pick_the_style_mode():
    models = make_models([ADAPTER, STYLE_ADAPTER])
    w = ComfyWorkflow(node_defs=models.node_inputs)
    model, clip, vae = w.add("CheckpointLoaderSimple", 3, ckpt_name=ANIMA)
    layer = workflow.Control.from_input(
        ControlInput(
            ControlMode.reference,
            Image.create(Extent(64, 64)),
            1.0,
            (0.0, 1.0),
            "style",
        )
    )

    model = workflow.apply_anima_references(
        w, model, [layer], workflow.Clip(clip, Arch.anima), vae, models
    )
    assert_serializable(w)

    (_, snapref) = nodes_of(w, "SnapRef")[0]
    assert snapref["inputs"]["adapter"] == STYLE_ADAPTER
    assert snapref["inputs"]["mode"] == "style"


def test_without_an_adapter_the_model_is_untouched():
    w, model = build([], count=1)

    assert not nodes_of(w, "SnapRef")
    assert str(model.node) == checkpoint_key(w, "CheckpointLoaderSimple")


def test_without_reference_layers_the_model_is_untouched():
    w, model = build([ADAPTER], count=0)

    assert not nodes_of(w, "SnapRef")
    assert str(model.node) == checkpoint_key(w, "CheckpointLoaderSimple")


def test_other_architectures_are_untouched():
    w, model = build([ADAPTER], count=1, arch=Arch.sdxl)

    assert not nodes_of(w, "SnapRef")
    assert str(model.node) == checkpoint_key(w, "CheckpointLoaderSimple")


def test_adapter_detection_ignores_unrelated_loras():
    assert resources.find_snapref_adapter(["detail_tweaker.safetensors"]) is None
    assert resources.find_snapref_adapter([]) is None
    assert resources.find_snapref_adapter([ADAPTER]) == ADAPTER
    assert resources.find_snapref_adapter([STYLE_ADAPTER]) == STYLE_ADAPTER


def test_the_widest_adapter_is_picked_without_asking():
    modes = {ADAPTER: ["cs", "char"], STYLE_ADAPTER: ["cs", "char", "style"]}

    # v2a sorts first by name, but only v2s can also serve a style reference.
    assert resources.find_snapref_adapter([ADAPTER, STYLE_ADAPTER], modes) == STYLE_ADAPTER
    assert resources.find_snapref_adapter([ADAPTER], modes) == ADAPTER
    assert resources.find_snapref_adapter([ADAPTER, STYLE_ADAPTER], {}) == STYLE_ADAPTER


def test_unknown_adapter_modes_do_not_block_the_layer():
    modes = {ADAPTER: ["cs", "char"]}

    assert resources.snapref_mode_supported(modes, ADAPTER, "char")
    assert not resources.snapref_mode_supported(modes, ADAPTER, "style")
    assert resources.snapref_mode_supported({}, ADAPTER, "style")
    assert resources.snapref_mode_supported({ADAPTER: []}, ADAPTER, "style")


def test_generate_graph_is_serializable_with_a_reference_layer():
    """Full builder path: the prompt is posted as JSON, so the Clip wrapper must not leak."""
    from ai_diffusion.backend.api import (
        CheckpointInput,
        ConditioningInput,
        SamplingInput,
    )
    from ai_diffusion.backend.resolution import ScaledExtent
    from ai_diffusion.files import FileFormat

    client_models = ClientModels()
    client_models.checkpoints = {
        ANIMA: CheckpointInfo(ANIMA, Arch.anima, FileFormat.diffusion)
    }
    client_models.resources = {
        resources.resource_id(ResourceKind.text_encoder, Arch.anima, "qwen_3_06b"): "te.safetensors",
        resources.resource_id(ResourceKind.vae, Arch.anima, "default"): "vae.safetensors",
    }
    client_models.loras = [ADAPTER]
    client_models.node_inputs = node_defs()
    models = ModelDict(client_models, ResourceKind.checkpoint, Arch.anima)

    sampling = SamplingInput("euler", "normal", 4.0, 20, 0, 1234)
    cond = workflow.Conditioning.from_input(ConditioningInput("a cat", ""), sampling)
    for weight in (1.0, 0.5):
        cond.control.append(
            workflow.Control.from_input(
                ControlInput(ControlMode.reference, Image.create(Extent(64, 64)), weight)
            )
        )

    w = ComfyWorkflow(node_defs=models.node_inputs)
    w = workflow.generate(
        w,
        CheckpointInput(ANIMA, Arch.anima),
        ScaledExtent.no_scaling(Extent(512, 512)),
        cond,
        sampling,
        workflow.MiscParams(1, 1, 0.0, 0.0),
        models,
    )

    assert_serializable(w)
    (_, snapref), *rest = nodes_of(w, "SnapRef")
    assert not rest, "a second reference layer must not add a second node"
    assert isinstance(snapref["inputs"]["clip"], list), "clip must be a node link, not a Clip"
    assert snapref["inputs"]["weight_2"] == pytest.approx(0.5)
