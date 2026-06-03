from fedprime.utils.env import add_vendor_paths

add_vendor_paths()
from Network.Models_Def.shufflenet import ShuffleNetG2  # noqa: E402,F401

__all__ = ["ShuffleNetG2"]

