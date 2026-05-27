"""
Integration wrapper: sequential patch processing with cross-patch state passing.

h_0^{s,m} = h_P^{s,m-1}  (final state of prev patch → init state of next)

Author: Xinyuan Zhang
"""

import torch
import torch.nn as nn
from .asd_ssm import AdaptiveScaleDecoupledMamba


class ASDSSMWrapper(nn.Module):
    """
    Wraps AdaptiveScaleDecoupledMamba for use inside SerializedAttention.

    Patches are processed **sequentially** so that the final hidden state of
    patch m is forwarded as the initial state of patch m+1, maintaining
    sequence continuity across patch boundaries.
    """

    def __init__(self, channels, num_scales=3):
        super().__init__()
        self.channels = channels
        self.num_scales = num_scales

        self.asd_mambas = nn.ModuleList([
            AdaptiveScaleDecoupledMamba(d_model=channels, num_scales=num_scales)
            for _ in range(num_scales)
        ])

    def forward(self, x, scale_id=0, x_res=None):
        """
        Args:
            x: (N_patches, L, C)
            scale_id: current scale index
            x_res: residual (passed through unchanged)
        Returns:
            x, x_res
        """
        scale_id = min(scale_id, self.num_scales - 1)
        mamba = self.asd_mambas[scale_id]
        N_patches = x.shape[0]

        h_fwd, h_bwd = None, None
        outputs = []
        for i in range(N_patches):
            out, h_fwd, h_bwd = mamba(x[i:i+1], scale_id, h_fwd, h_bwd)
            outputs.append(out)

        x = torch.cat(outputs, dim=0)
        return x, x_res
