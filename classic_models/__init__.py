"""Model registry for the DiSFormer paper release."""

from importlib import import_module

_MODEL_SPECS = {
    "disformer": (".disformer", "disformer_base_patch16_224"),
    "vit_baseline": (".vision_transformer", "vit_base_patch16_224"),
    "vision_transformer_small": (".vision_transformer", "vit_base_patch32_224"),
    "vision_transformer2": (".vision_transformer", "vit_base_patch16_224_in21k"),
    "vision_transformer_big": (".vision_transformer", "vit_large_patch16_224"),
    "alexnet": (".alexnet", "alexnet"),
    "vgg": (".vggnet", "vgg16"),
    "resnet": (".resnet", "resnet50"),
    "densenet": (".densenet", "densenet169"),
    "mobilenet_v3": (".mobilenet_v3", "mobilenet_v3_small"),
    "mobilenet_v3_large": (".mobilenet_v3", "mobilenet_v3_large"),
    "efficient_v2_small": (".efficientnet_v2", "efficientnetv2_s"),
    "efficient_v2": (".efficientnet_v2", "efficientnetv2_m"),
    "efficient_v2_large": (".efficientnet_v2", "efficientnetv2_l"),
    "inceptionnext": (".inceptionnext", "inceptionnext_base"),
    "van": (".van", "VAN"),
}


def _load_constructor(model_name):
    if model_name not in _MODEL_SPECS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {sorted(_MODEL_SPECS)}"
        )
    module_name, constructor_name = _MODEL_SPECS[model_name]
    module = import_module(module_name, package=__name__)
    return getattr(module, constructor_name)


def find_model_using_name(model_name, num_classes, pretrained=False):
    constructor = _load_constructor(model_name)
    if model_name in {"alexnet", "resnet", "vgg"}:
        return constructor(num_classes=num_classes, pretrained=pretrained)
    return constructor(num_classes=num_classes)


def disformer_base_patch16_224(num_classes=1000):
    return _load_constructor("disformer")(num_classes=num_classes)
