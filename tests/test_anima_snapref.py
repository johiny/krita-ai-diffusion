"""Anima reference layers are routed through the SnapRef node.

SnapRef keeps the reference K/V of every attention block for the sampler, so a reference
control layer on an Anima checkpoint has to wrap the model instead of the conditioning.
"""

from ai_diffusion.backend import resources, workflow
from ai_diffusion.backend.api import ControlInput
from ai_diffusion.backend.client import ClientModels, ModelDict
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


def make_models(loras: list[str], arch: Arch = Arch.anima) -> ModelDict:
    client_models = ClientModels()
    client_models.loras = list(loras)
    client_models.node_inputs = node_defs()
    return ModelDict(client_models, ResourceKind.checkpoint, arch)


def reference_layers(count: int, strength: float = 1.0) -> list[workflow.Control]:
    return [
        workflow.Control.from_input(
            ControlInput(ControlMode.reference, Image.create(Extent(64, 64)), strength)
        )
        for _ in range(count)
    ]


def build(loras: list[str], count: int, arch: Arch = Arch.anima, strength: float = 1.0):
    models = make_models(loras, arch)
    w = ComfyWorkflow(node_defs=models.node_inputs)
    model, clip, vae = w.add("CheckpointLoaderSimple", 3, ckpt_name=ANIMA)
    model = workflow.apply_anima_references(
        w, model, reference_layers(count, strength), clip, vae, models
    )
    return w, model


def nodes_of(w: ComfyWorkflow, class_type: str):
    return [(key, node) for key, node in w.root.items() if node["class_type"] == class_type]


def test_reference_layer_wraps_the_model_with_snapref():
    w, model = build([ADAPTER], count=1)

    (key, node), *rest = nodes_of(w, "SnapRef")
    assert not rest
    assert node["inputs"]["adapter"] == ADAPTER
    assert node["inputs"]["weight_1"] == 1.0
    assert node["inputs"]["mode"] == "cs"
    (_, checkpoint) = nodes_of(w, "CheckpointLoaderSimple")[0]
    assert node["inputs"]["clip"] == [checkpoint_key(w, "CheckpointLoaderSimple"), 1]
    assert str(model.node) == key, "the sampler must read the SnapRef output"


def checkpoint_key(w: ComfyWorkflow, class_type: str) -> str:
    return nodes_of(w, class_type)[0][0]


def test_reference_image_comes_from_the_control_layer():
    w, _ = build([ADAPTER], count=1)

    (_, snapref), *rest = nodes_of(w, "SnapRef")
    (_, load_image), *other = nodes_of(w, "ETN_LoadImageCache")
    assert not rest and not other
    assert snapref["inputs"]["image"] == [checkpoint_key(w, "ETN_LoadImageCache"), 0]


def test_layer_strength_maps_to_the_linear_weight():
    w, _ = build([ADAPTER], count=1, strength=0.5)

    (_, snapref) = nodes_of(w, "SnapRef")[0]
    assert snapref["inputs"]["weight_1"] == 0.5


def test_several_reference_layers_chain():
    w, model = build([ADAPTER], count=2)

    first, second = nodes_of(w, "SnapRef")
    assert second[1]["inputs"]["model"] == [first[0], 0]
    assert str(model.node) == second[0]


def test_character_adapter_wins_over_the_style_one():
    w, _ = build([STYLE_ADAPTER, ADAPTER], count=1)

    (_, snapref) = nodes_of(w, "SnapRef")[0]
    assert snapref["inputs"]["adapter"] == ADAPTER


def test_explicit_adapter_and_mode_override_the_defaults():
    models = make_models([ADAPTER, STYLE_ADAPTER])
    w = ComfyWorkflow(node_defs=models.node_inputs)
    model, clip, vae = w.add("CheckpointLoaderSimple", 3, ckpt_name=ANIMA)
    layer = workflow.Control.from_input(
        ControlInput(
            ControlMode.reference,
            Image.create(Extent(64, 64)),
            1.0,
            (0.0, 1.0),
            STYLE_ADAPTER,
            "style",
        )
    )

    model = workflow.apply_anima_references(w, model, [layer], clip, vae, models)

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
