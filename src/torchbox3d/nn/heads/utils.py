def cuboids_to_vertices_r3(strided_regressands: Tensor):
    B, _, H, W = strided_regressands.shape
    cuboids = strided_regressands.view(B, -1, 8, H, W)
    pose = cuboids[:, :, -2:].double()
    # pose = cuboids[:, :, -2:].permute(0, 1, 3, 4, 2)
    # rots = rotation_6d_to_matrix(pose).permute(0, 1, 4, 5, 2, 3)
    pose = F.normalize(pose, dim=2)
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
    dims = cuboids[:, :, 3:6].exp()

    verts_obj = dims[:, :, None, :] / 2 * unit_verts[None, None, :, :, None, None]

    sin = pose[:, :, None, 0:1]
    cos = pose[:, :, None, 1:2]
    xctr = cos * verts_obj[:, :, :, 0:1] - sin * verts_obj[:, :, :, 1:2]
    yctr = sin * verts_obj[:, :, :, 0:1] + cos * verts_obj[:, :, :, 1:2]
    zctr = verts_obj[:, :, :, 2:3]

    xctr = xctr + cart[:, :, None, 0:1]
    yctr = yctr + cart[:, :, None, 1:2]
    zctr = zctr + cart[:, :, None, 2:3]
    verts_ego = torch.concatenate([xctr, yctr, zctr], dim=3)

    # verts_ego = (
    #     torch.einsum("bnijhw,bnvjhw->bnvihw", rots, verts_obj) + cart[:, :, None]
    # )
    theta = torch.atan2(cuboids[:, :, 6:7], cuboids[:, :, 7:8])
    cuboids = torch.concatenate([cart, dims, theta], dim=2)
    return verts_ego.permute(0, 1, 4, 5, 2, 3).flatten(1, 3), cuboids.permute(
        0, 1, 3, 4, 2
    ).flatten(1, 3)


def boxes_to_rot(bbox):
    return torch.stack(
        (
            bbox[..., 0] - bbox[..., 3] / 2,
            bbox[..., 1] - bbox[..., 4] / 2,
            bbox[..., 2] - bbox[..., 5] / 2,
            bbox[..., 0] + bbox[..., 3] / 2,
            bbox[..., 1] + bbox[..., 4] / 2,
            bbox[..., 2] + bbox[..., 5] / 2,
        ),
        dim=-1,
    )


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def cuboids_to_vertices(strided_regressands: Tensor):
    B, _, H, W = strided_regressands.shape
    cuboids = strided_regressands.view(B, -1, 8, H, W)
    pose = cuboids[:, :, -2:].double()
    # pose = cuboids[:, :, -2:].permute(0, 1, 3, 4, 2)
    # rots = rotation_6d_to_matrix(pose).permute(0, 1, 4, 5, 2, 3)
    pose = F.normalize(pose, dim=2)
    unit_verts = torch.as_tensor(
        [
            [+1, +1],  # 3
            [-1, +1],  # 2
            [-1, -1],  # 6
            [+1, -1],  # 7
        ],
        dtype=cuboids.dtype,
        device=cuboids.device,
    )

    cart = cuboids[:, :, :3]
    dims = cuboids[:, :, 3:6].exp()

    verts_obj = dims[:, :, None, :2] / 2 * unit_verts[None, None, :, :, None, None]

    sin = pose[:, :, None, 0:1]
    cos = pose[:, :, None, 1:2]
    xctr = cos * verts_obj[:, :, :, 0:1] - sin * verts_obj[:, :, :, 1:2]
    yctr = sin * verts_obj[:, :, :, 0:1] + cos * verts_obj[:, :, :, 1:2]

    xctr += cart[:, :, None, 0:1]
    yctr += cart[:, :, None, 1:2]
    verts_ego = torch.concatenate([xctr, yctr], dim=3)

    # verts_ego = (
    #     torch.einsum("bnijhw,bnvjhw->bnvihw", rots, verts_obj) + cart[:, :, None]
    # )
    # theta = torch.atan2(cuboids[:, :, 6:7], cuboids[:, :, 7:8])
    cuboids = torch.concatenate([cart, dims], dim=2)
    return verts_ego.permute(0, 1, 4, 5, 2, 3).flatten(1, 3), cuboids.permute(
        0, 1, 3, 4, 2
    ).flatten(1, 3)
