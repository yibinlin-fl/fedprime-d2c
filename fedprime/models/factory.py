from __future__ import annotations

from fedprime.utils.env import add_vendor_paths


def build_models(model_names: list[str], num_classes: int):
    add_vendor_paths()
    from Dataset.utils import init_nets

    return init_nets(
        n_parties=len(model_names),
        nets_name_list=model_names,
        num_classes=num_classes,
    )


def forward_logits(model, images):
    output = model(images)
    if isinstance(output, tuple):
        return output[0]
    return output

