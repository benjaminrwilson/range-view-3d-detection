import torch.nn.functional as F
from torch import Tensor


def depth_saliency(cart: Tensor, mask: Tensor, num_neighbors: int = 3) -> Tensor:
    B, _, H, W = cart.shape
    padding = num_neighbors // 2
    cartesian_coordinates = F.unfold(cart, num_neighbors, padding=padding).view(
        B, -1, num_neighbors**2, H * W
    )

    original_mask = mask.clone()

    mask = F.unfold(mask.type_as(cart), num_neighbors, padding=padding).view(
        B, -1, num_neighbors**2, H * W
    )

    cartesian_anchor = cartesian_coordinates[:, :, 4:5]
    relative_coordinates = cartesian_coordinates - cartesian_anchor
    dists = relative_coordinates.norm(dim=1, keepdim=True)
    # dists = relative_coordinates.norm(
    #     dim=1, keepdim=True
    # ) * mask + mask.logical_not().masked_fill(mask.logical_not(), torch.nan)
    # dists, _ = torch.nanmedian(dists, dim=2, keepdim=True)
    # dists = torch.nan_to_num(dists, nan=0.0)

    max_dists, _ = (dists * mask).max(dim=2, keepdim=True)
    # max_dists /= cartesian_anchor.norm(dim=1, keepdim=True)
    # max_dists = torch.nan_to_num(max_dists, posinf=0.0)
    return max_dists.view(B, -1, H, W) * original_mask


def compute_cartesian_offsets(
    cart: Tensor, mask: Tensor, num_neighbors: int = 3
) -> Tensor:
    B, _, H, W = cart.shape
    padding = num_neighbors // 2
    cartesian_coordinates = F.unfold(cart, num_neighbors, padding=padding).view(
        B, -1, num_neighbors**2, H * W
    )

    mask = F.unfold(mask.type_as(cart), num_neighbors, padding=padding).view(
        B, -1, num_neighbors**2, H * W
    )

    cartesian_anchor = cartesian_coordinates[:, :, 4:5]
    relative_coordinates = (cartesian_coordinates - cartesian_anchor) * mask
    return relative_coordinates.view(B, -1, H, W)
