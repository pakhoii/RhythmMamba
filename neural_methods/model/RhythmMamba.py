""" 
RhythmMamba: Fast Remote Physiological Measurement with Arbitrary Length Videos
"""
import torch
from torch import nn
import torch.nn.functional as F
import torch.fft
from functools import partial
from timm.models.layers import trunc_normal_, lecun_normal_
from timm.models.layers import DropPath, to_2tuple
import math
from einops import rearrange
from mamba_ssm.modules.mamba_simple import Mamba


class TemporalNorm(nn.Module):
    """Temporal Normalization Block from: https://arxiv.org/html/2411.15283
        
        Detrending step:
            P_detrended = (I - X (X^T X)^{-1} X^T) P = D @ P
             
        RMS normalization step:
            P_norm = P_detrended / sqrt(mean(P_detrended^2) + eps)
    """

    def __init__(self, eps=1e-6):
        super(TemporalNorm, self).__init__()
        self.eps = eps
        self._cache = {} # key: (T, device, dtype), value: D matrix

    def _get_detrend_matrix(self, T, device, dtype):
        key = (T, device.type, dtype)
        if key in self._cache:
            return self._cache[key]

        t = torch.arange(T, dtype=dtype, device=device).unsqueeze(1)
        ones = torch.ones(T, 1, dtype=dtype, device=device)
        X = torch.cat([ones, t], dim=1)

        XtX_inv = torch.linalg.pinv(X.T @ X)
        H = X @ XtX_inv @ X.T

        D = torch.eye(T, dtype=dtype, device=device) - H
        self._cache[key] = D
        
        return D
        
    def forward(self, x):
        N, D, C, H, W = x.shape
        
        A = self._get_detrend_matrix(D, x.device, x.dtype)  # [D, D]
        
        # Detrending
        x_flat = x.permute(1, 0, 2, 3, 4).reshape(D, -1)   # [D, N*C*H*W]
        x_detrended = A @ x_flat                             # [D, N*C*H*W]
        
        # RMS normalization
        rms = torch.sqrt(
            torch.mean(x_detrended ** 2, dim=0, keepdim=True) + self.eps
        )
        x_normalized = x_detrended / rms
        
        x_normalized = x_normalized.reshape(D, N, C, H, W).permute(1, 0, 2, 3, 4)
        return x_normalized
    

class AdaptiveStem(nn.Module):
    def __init__(self, channels=3, eps=1e-6):
        super(AdaptiveStem, self).__init__()
        self.temporal_norm = TemporalNorm(eps=eps)
        
    def forward(self, x):
        x = self.temporal_norm(x)
        return x


class TemporalShift(nn.Module):
    """
        Temporal Shift Module from: https://arxiv.org/pdf/2006.03790 
        Combined with Fusion_Stem design logic
    """
    def __init__(self, fold_div=3, dim=24):
        super(TemporalShift, self).__init__()
        self.fold_div = fold_div
        self.dim = dim
        
        self.stem1 = nn.Sequential(
            nn.Conv2d(3, dim//2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim//2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )
        
        self.stem2 = nn.Sequential(
            nn.Conv2d(dim//2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )
        
    def forward(self, x):
        N, D, C_in, H_in, W_in = x.shape
        
        x = x.reshape(N * D, C_in, H_in, W_in)
        x = self.stem1(x)

        _, C_new, H_new, W_new = x.shape
        
        x = x.reshape(N, D, C_new, H_new, W_new)
        out = x.clone()
        
        fold = C_new // self.fold_div

        # Shift forward (t+1 -> t)
        out[:, :-1, :fold, :, :] = x[:, 1:, :fold, :, :]
        
        # Shift backward (t-1 -> t)
        out[:, 1:, fold:2*fold, :, :] = x[:, :-1, fold:2*fold, :, :]
        
        # Keep the rest unchanged
        # ...
            
        out = out.view(N * D, C_new, H_new, W_new)
        # out = self.se(out) * out
        out = self.stem2(out)
        
        return out
    

class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=3, keepdim=True)
        xsum = torch.sum(xsum, dim=4, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[3] * xshape[4] * 0.5

    def get_config(self):
        """May be generated manually. """
        config = super(Attention_mask, self).get_config()
        return config


class Frequencydomain_FFN(nn.Module):
    def __init__(self, dim, mlp_ratio):
        super().__init__()

        self.scale = 0.02
        self.dim = dim * mlp_ratio

        self.r = nn.Parameter(self.scale * torch.randn(self.dim, self.dim))
        self.i = nn.Parameter(self.scale * torch.randn(self.dim, self.dim))
        self.rb = nn.Parameter(self.scale * torch.randn(self.dim))
        self.ib = nn.Parameter(self.scale * torch.randn(self.dim))

        self.fc1 = nn.Sequential(
            nn.Conv1d(dim, dim * mlp_ratio, 1, 1, 0, bias=False),  
            nn.BatchNorm1d(dim * mlp_ratio),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(dim * mlp_ratio, dim, 1, 1, 0, bias=False),  
            nn.BatchNorm1d(dim),
        )


    def forward(self, x):
        B, N, C = x.shape
  
        x = self.fc1(x.transpose(1, 2)).transpose(1, 2)

        x_fre = torch.fft.fft(x, dim=1, norm='ortho') # FFT on N dimension

        x_real = F.relu(
            torch.einsum('bnc,cc->bnc', x_fre.real, self.r) - \
            torch.einsum('bnc,cc->bnc', x_fre.imag, self.i) + \
            self.rb
        )
        x_imag = F.relu(
            torch.einsum('bnc,cc->bnc', x_fre.imag, self.r) + \
            torch.einsum('bnc,cc->bnc', x_fre.real, self.i) + \
            self.ib
        )

        x_fre = torch.stack([x_real, x_imag], dim=-1).float()
        x_fre = torch.view_as_complex(x_fre)
        x = torch.fft.ifft(x_fre, dim=1, norm="ortho")
        x = x.to(torch.float32)

        x = self.fc2(x.transpose(1, 2)).transpose(1, 2)
        return x


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=48, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  
            d_state=d_state,  
            d_conv=d_conv, 
            expand=expand  
        )
    def forward(self, x):
        B, N, C = x.shape
        x_norm = self.norm(x)
        x_mamba = self.mamba(x_norm)    
        return x_mamba


class Block_mamba(nn.Module):
    def __init__(self, 
        dim, 
        mlp_ratio,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = MambaLayer(dim)
        self.mlp = Frequencydomain_FFN(dim,mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, D, C = x.size()
        #Multi-temporal Parallelization
        path = 3
        segment = 2**(path-1)
        tt = D // segment
        x_r = x.repeat(segment,1,1)
        x_o = x_r.clone()
        for i in range(1,segment):
            x_o[i*B:(i+1)*B,:D-i*tt,:] = x_r[i*B:(i+1)*B,i*tt:,:]
        x_o = self.attn(x_o)
        for i in range(1,segment):
            for j in range(i):
                x_o[0:B, tt*i: tt*(i+1) , :] = x_o[0:B, tt*i: tt*(i+1) , :] + x_o[B*(j+1):B*(j+2), tt*(i-j-1): tt*(i-j) , :]
            x_o[0:B, tt*i: tt*(i+1) , :] = x_o[0:B, tt*i: tt*(i+1) , :] / (i+1)
        x = x + self.drop_path(self.norm1(x_o[0:B]))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class RhythmMamba(nn.Module):
    def __init__(self, 
                 depth=24, 
                 embed_dim=96, 
                 mlp_ratio=2,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 initializer_cfg=None,
                 device=None,
                 dtype=None,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 
        super().__init__()
        self.embed_dim = embed_dim
        
        self.Adaptive_Stem = AdaptiveStem()
        
        # self.Fusion_Stem = Fusion_Stem(dim=embed_dim//4)
        self.TemporalShift = TemporalShift(dim=embed_dim//4)
        self.attn_mask = Attention_mask()

        self.stem3 = nn.Sequential(
            nn.Conv3d(embed_dim//4, embed_dim, kernel_size=(2, 5, 5), stride=(2, 1, 1),padding=(0,2,2)),
            nn.BatchNorm3d(embed_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.blocks = nn.ModuleList([Block_mamba(
            dim = embed_dim, 
            mlp_ratio = mlp_ratio,
            drop_path=inter_dpr[i], 
            norm_layer=nn.LayerNorm,)
        for i in range(depth)])

        self.upsample = nn.Upsample(scale_factor=2)
        self.ConvBlockLast = nn.Conv1d(embed_dim, 1, kernel_size=1,stride=1, padding=0)

        # init
        self.apply(segm_init_weights)
        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )


    def forward(self, x):
        B, D, C, H, W = x.shape
        
        # Adaptive Stem
        x = self.Adaptive_Stem(x)

        x = self.TemporalShift(x)  #[N*D C H/8 W/8]
        x = x.view(B,D,self.embed_dim//4,H//8,W//8).permute(0,2,1,3,4)
        x = self.stem3(x)

        mask = torch.sigmoid(x)
        mask = self.attn_mask(mask)
        x = x * mask

        x = torch.mean(x,4)
        x = torch.mean(x,3)
        x = rearrange(x, 'b c t -> b t c')

        for blk in self.blocks:
            x = blk(x)

        rPPG = x.permute(0,2,1) 
        rPPG = self.upsample(rPPG)
        rPPG = self.ConvBlockLast(rPPG)    #[N, 1, D]
        rPPG = rPPG.squeeze(1)

        return rPPG