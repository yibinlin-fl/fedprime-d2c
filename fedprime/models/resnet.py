from fedprime.utils.env import add_vendor_paths

add_vendor_paths()
from Network.Models_Def.resnet import ResNet10, ResNet12  # noqa: E402,F401

__all__ = ["ResNet10", "ResNet12"]

