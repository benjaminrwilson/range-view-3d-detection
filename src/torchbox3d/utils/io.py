"""Input / output methods."""

import torch
from torch import Tensor
from torchvision.io import decode_jpeg, decode_png, read_file, write_jpeg, write_png


@torch.jit.script
def read_img(path: str, device: str = "cpu") -> Tensor:
    """Read an image into memory.

    NOTE: Path variable is currently type `str` instead of `Path`
        to allow for JIT compilation.

    Args:
        path: File path.
        device: Device to support jpeg decoding.

    Returns:
        (B,C,H,W) The image tensor.

    Raises:
        ValueError: If the file extension is not a supported image format.
    """
    ext = path.split(".")[-1].lower()
    img = read_file(path)
    if ext == "jpg":
        return decode_jpeg(img, device=device)  # type: ignore
    elif ext == "png":
        return decode_png(img).to(device=device)  # type: ignore
    else:
        raise ValueError("Invalid image type!")


@torch.jit.script
def write_img(img: Tensor, path: str) -> None:
    """Write an image to disk.

    NOTE: Path variable is currently type `str` instead of `Path`
        to allow for JIT compilation.

    Args:
        img: (C,H,W) image to write.
        path: File path.

    Raises:
        ValueError: If the file extension is not a supported image format.
    """
    ext = path.split(".")[-1].lower()
    if ext == "jpg":
        write_jpeg(img, path)
    elif ext == "png":
        write_png(img, path)
    else:
        raise ValueError("Invalid image type!")
