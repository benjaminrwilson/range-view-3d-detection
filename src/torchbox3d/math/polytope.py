"""Geometric methods for polytopes.

Reference: https://en.wikipedia.org/wiki/Polytope
"""

import torch
from kornia.geometry.conversions import (
    quaternion_from_euler,
    quaternion_to_rotation_matrix,
)
from torch import Tensor


@torch.jit.script
def compute_interior_points_mask(points_xyz: Tensor, cuboid_vertices: Tensor) -> Tensor:
    r"""Compute the interior points within a set of _axis-aligned_ cuboids.

    Reference:
        https://math.stackexchange.com/questions/1472049/check-if-a-point-is-inside-a-rectangular-shaped-area-3d

            5------4
            |\\    |\\
            | \\   | \\
            6--\\--7  \\
            \\  \\  \\ \\
        l    \\  1-------0    h
         e    \\ ||   \\ ||   e
          n    \\||    \\||   i
           g    \\2------3    g
            t      width.     h
             h.               t.

    Args:
        points_xyz: (N,3) Points in Cartesian space.
        cuboid_vertices: (K,8,3) Vertices of the cuboids.

    Returns:
        (N,) A tensor of boolean flags indicating whether the points
            are interior to the cuboid.
    """
    vertices = cuboid_vertices[:, [6, 3, 1]]
    uvw = cuboid_vertices[:, 2:3] - vertices
    reference_vertex = cuboid_vertices[:, 2:3]

    dot_uvw_reference = uvw @ reference_vertex.transpose(1, 2)
    dot_uvw_vertices = torch.diagonal(uvw @ vertices.transpose(1, 2), 0, 2)[..., None]
    dot_uvw_points = uvw @ points_xyz.T

    constraint_a = torch.logical_and(
        dot_uvw_reference <= dot_uvw_points, dot_uvw_points <= dot_uvw_vertices
    )
    constraint_b = torch.logical_and(
        dot_uvw_reference >= dot_uvw_points, dot_uvw_points >= dot_uvw_vertices
    )
    is_interior: Tensor = torch.logical_or(constraint_a, constraint_b).all(dim=1)
    return is_interior


@torch.jit.script
def compute_polytope_interior(
    points_xyz: Tensor,
    polytope: Tensor,
) -> Tensor:
    """Compute the interior points within a set of polytopes.

    Args:
        points_xyz: (N,3) Points in Cartesian space.
        polytope: (K,8,3) Vertices of the polytope.

    Returns:
        (K,) Booleans indicating whether the points are interior to the cuboid.
    """
    return compute_interior_points_mask(polytope, points_xyz)


def cuboids_to_vertices(cuboids: Tensor):
    assert cuboids.shape[0] > 0
    B, K, C = cuboids.shape
    unit_verts = torch.as_tensor(
        [
            [+1, +1, +1],  # 0
            [+1, -1, +1],  # 1
            [+1, -1, -1],  # 2
            [+1, +1, -1],  # 3
            [-1, +1, +1],  # 4
            [-1, -1, +1],  # 5
            [-1, -1, -1],  # 6
            [-1, +1, -1],  # 7
        ],
        dtype=cuboids.dtype,
        device=cuboids.device,
    )

    cart = cuboids[:, :, :3]
    dims = cuboids[:, :, 3:6]
    yaw = cuboids[:, :, -1:]

    pitch = torch.zeros_like(yaw)
    roll = torch.zeros_like(yaw)

    quat = torch.concatenate(quaternion_from_euler(roll, pitch, yaw), dim=-1)
    rots = quaternion_to_rotation_matrix(quat.reshape(B * K, 4)).reshape(B, K, 3, 3)

    verts_obj = dims[:, :, None] / 2.0 * unit_verts[None, None]
    # verts_ego = torch.einsum("dbij,dbkj->dbki", rots, verts_obj) + cart[:, :, None]
    verts_ego = verts_obj @ rots.transpose(3, 2) + cart[:, :, None]
    return verts_ego
