"""
Adaptive Scale-Decoupled State Space Model (ASD-SSM)

Core: per-patch, scale-aware state transition parameter generation,
with scale constraint factors controlling hierarchical decay rates.

Recurrence: h_k = diag(Ā_s) · h_{k-1} + B_k · u_k,  y_k = C_k^T · h_k
where Ā_s = α_s · σ(λ · MLP([f̃_global ∥ e_s]))

Author: Xinyuan Zhang
"""

import torch
import torch.nn as nn
from mamba_ssm.ops.triton.layernorm import RMSNorm

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_CUDA_SCAN = True
except ImportError:
    HAS_CUDA_SCAN = False


class ScaleAwareParameterGenerator(nn.Module):
    """
    Scale-aware parameter generator for ASD-SSM.

    Eq.(1): f̃_global = MLP_global(GAP(patch_m))  ∈ R^{C/4}
    Eq.(2): Â_raw   = σ(λ · MLP([f̃_global ∥ e_s]))
    Eq.(3): Ā_{s,m} = α_s · Â_raw                 ∈ [0, α_s]
    """

    def __init__(self, d_model, num_scales=3, expand=2, use_global_feature=True):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(d_model * expand)
        self.num_scales = num_scales
        self.use_global_feature = use_global_feature
        self.temperature = 0.1  # λ

        if use_global_feature:
            self.global_feature_extractor = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
            )
            input_dim = d_model // 2
        else:
            input_dim = d_model // 4

        self.scale_embedding = nn.Embedding(num_scales, d_model // 4)

        self.param_generator = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.d_inner),
        )

        # α_1=0.3 (finest) … α_S=0.9 (coarsest), monotonically increasing
        self.register_buffer(
            'scale_constraints',
            torch.linspace(0.3, 0.9, num_scales)
        )

    def forward(self, features, scale_id):
        """
        Args:
            features: (B, L, C)  patch features
            scale_id: int        scale index (0 = finest, S-1 = coarsest)
        Returns:
            A_bar: (B, D_inner)  ∈ [0, α_s]
        """
        B, L, C = features.shape
        device = features.device

        if self.use_global_feature:
            global_feat = self.global_feature_extractor(
                features.transpose(1, 2)
            )
        else:
            global_feat = None

        scale_emb = self.scale_embedding(
            torch.tensor([scale_id], device=device, dtype=torch.long)
        ).expand(B, -1)

        if self.use_global_feature:
            condition = torch.cat([global_feat, scale_emb], dim=-1)
        else:
            condition = scale_emb

        A_raw = self.param_generator(condition)
        alpha_s = self.scale_constraints[scale_id]
        A_bar = alpha_s * torch.sigmoid(self.temperature * A_raw)

        return A_bar


class SelectiveSSMCore(nn.Module):
    """
    Single-direction SSM whose state transition matrix Ā is supplied externally.

    h_k = diag(Ā) · h_{k-1} + B_k · u_k
    y_k = C_k^T · h_k

    B, C follow the standard Mamba selective mechanism (linear projections).
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner,
        )
        self.act = nn.SiLU()

        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    # ------------------------------------------------------------------
    def _scan_sequential(self, u, A, B, C, h_init):
        """Reference implementation (sequential). Equivalent to parallel scan."""
        Bsz, L, D = u.shape
        N = self.d_state
        A_exp = A.unsqueeze(-1).expand(-1, -1, N)
        h = h_init if h_init is not None else u.new_zeros(Bsz, D, N)
        ys = []
        for k in range(L):
            h = A_exp * h + B[:, k].unsqueeze(1) * u[:, k].unsqueeze(-1)
            ys.append((C[:, k].unsqueeze(1) * h).sum(-1))
        return torch.stack(ys, dim=1), h

    def _scan_cuda(self, u, A_bar, B, C, h_init):
        """
        CUDA-accelerated parallel scan via mamba_ssm.
        Convert Ā to log-space so that  exp(Δ·A_log) = Ā  when Δ ≡ 1.
        """
        Bsz, L, D = u.shape
        A_log = torch.log(A_bar.clamp(min=1e-6))
        A_log_2d = A_log.unsqueeze(-1).expand(-1, -1, self.d_state)
        delta = u.new_ones(Bsz, D, L)
        y = selective_scan_fn(
            u.transpose(1, 2).contiguous(),
            delta,
            A_log_2d,
            B.transpose(1, 2).contiguous(),
            C.transpose(1, 2).contiguous(),
            D=None, z=None,
            delta_bias=None, delta_softplus=False,
        )
        return y.transpose(1, 2), None

    # ------------------------------------------------------------------
    def forward(self, x, A_bar, h_init=None):
        """
        Args:
            x:      (B, L, D_model)
            A_bar:  (B, D_inner)      from ScaleAwareParameterGenerator, ∈ [0, α_s]
            h_init: (B, D_inner, N)   optional initial hidden state
        Returns:
            y:       (B, L, D_model)
            h_final: (B, D_inner, N)
        """
        L = x.shape[1]
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        x_branch = self.conv1d(x_branch.transpose(1, 2))[..., :L].transpose(1, 2)
        x_branch = self.act(x_branch)

        B = self.B_proj(x_branch)
        C = self.C_proj(x_branch)

        A_inner = A_bar

        if HAS_CUDA_SCAN and not torch.is_grad_enabled():
            y, h_final = self._scan_cuda(x_branch, A_inner, B, C, h_init)
        else:
            y, h_final = self._scan_sequential(x_branch, A_inner, B, C, h_init)

        y = y * self.act(z)
        y = self.out_proj(y)
        return y, h_final


class BidirectionalASDSSM(nn.Module):
    """
    Bidirectional SSM: forward and backward directions share the same Ā_s.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.fwd_ssm = SelectiveSSMCore(d_model, d_state, d_conv, expand)
        self.bwd_ssm = SelectiveSSMCore(d_model, d_state, d_conv, expand)
        self.merge = nn.Linear(2 * d_model, d_model, bias=False)

    def forward(self, x, A_bar, h_fwd_init=None, h_bwd_init=None):
        y_fwd, h_fwd = self.fwd_ssm(x, A_bar, h_fwd_init)
        y_bwd, h_bwd = self.bwd_ssm(x.flip(1), A_bar, h_bwd_init)
        y_bwd = y_bwd.flip(1)
        y = self.merge(torch.cat([y_fwd, y_bwd], dim=-1))
        return y, h_fwd, h_bwd


class AdaptiveScaleDecoupledMamba(nn.Module):
    """
    ASD-Mamba block for a single scale.

    1. ScaleAwareParameterGenerator  → per-patch Ā_s
    2. BidirectionalASDSSM           → SSM recurrence with custom Ā_s
    3. Cross-patch state passing     → h_0^{s,m} = h_P^{s,m-1}
    """

    def __init__(self, d_model, num_scales=3, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.param_gen = ScaleAwareParameterGenerator(d_model, num_scales, expand)
        self.bi_ssm = BidirectionalASDSSM(d_model, d_state, d_conv, expand)
        self.norm = RMSNorm(d_model, eps=1e-5)

    def forward(self, x, scale_id, h_fwd=None, h_bwd=None):
        """
        Args:
            x:        (1, L, C) single-patch features
            scale_id: int
            h_fwd, h_bwd: hidden states from previous patch (state passing)
        Returns:
            y, h_fwd_out, h_bwd_out
        """
        residual = x
        x = self.norm(x)
        A_bar = self.param_gen(x, scale_id)
        y, h_fwd_out, h_bwd_out = self.bi_ssm(x, A_bar, h_fwd, h_bwd)
        return y + residual, h_fwd_out, h_bwd_out
