from fedprime.utils.env import add_vendor_paths

add_vendor_paths()
from utils.diffeomorphism import Diffeo  # noqa: E402,F401

__all__ = ["Diffeo"]

