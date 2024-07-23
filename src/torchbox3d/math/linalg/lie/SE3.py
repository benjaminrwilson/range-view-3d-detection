"""SE(3) group class."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import torch
from torch import Tensor


@dataclass
class SE3:
    """SE(3) lie group object.

    References:
        http://ethaneade.com/lie_groups.pdf

    Args:
        R: (3,3) 3D rotation matrix.
        t: (3,) 3D translation vector.
    """

    R: Tensor
    t: Tensor

    @cached_property
    def Rt(self) -> Tensor:
        """Return the (4,4) homogeneous transformation matrix."""
        Rt: Tensor = torch.eye(4)
        Rt[:3, :3] = self.R
        Rt[:3, 3] = self.t
        return Rt

    def transform_from(self, points_xyz: Tensor) -> Tensor:
        """Apply the SE(3) transformation to the points.

        Args:
            points_xyz: (N,3) Tensor of points.

        Returns:
            (N,3) The transformed tensor of points.
        """
        return points_xyz @ self.R.T + self.t

    def inverse(self) -> SE3:
        """Return the inverse of the current SE(3) transformation.

        Returns:
            The SE(3) transformation: target_SE3_src.
        """
        return SE3(R=self.R.T, t=self.R.T.dot(-self.t))

    def compose(self, right_SE3: SE3) -> SE3:
        """Right multiply this class' transformation matrix T with an SE(3) instance.

        Algebraic representation: chained_se3 = T * right_SE3

        Args:
            right_SE3: Another instance of SE3 class.

        Returns:
            The SE(3) composition.
        """
        Rt: Tensor = self.Rt @ right_SE3.Rt
        composed_SE3 = SE3(
            R=Rt[:3, :3],
            t=Rt[:3, 3],
        )
        return composed_SE3
