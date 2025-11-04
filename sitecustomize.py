# sitecustomize.py
import os, math, torch
import torch.nn as nn
from torch.autograd import Function

_USE_COMPILE = os.environ.get("LORA_NS_NO_COMPILE", "0") != "1"

def _zeropower_via_newtonschulz5_impl(G, steps: int):
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    transposed = False
    if G.size(-2) > G.size(-1):
        X = X.mT
        transposed = True
    # compute NS in fp32 for stability under AMP
    X32 = X.to(torch.float32)
    X32 = X32 / (X32.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X32 @ X32.mT
        B = b * A + c * (A @ A)
        X32 = a * X32 + B @ X32
    X = X32.to(G.dtype)
    if transposed:
        X = X.mT
    return X

zeropower_via_newtonschulz5 = (
    torch.compile(_zeropower_via_newtonschulz5_impl) if _USE_COMPILE else _zeropower_via_newtonschulz5_impl
)

class LinearFunction(Function):
    """
    y = x @ (W^T + s * U V) + b,  s = alpha / r   (alpha semantics match PEFT)
    Saves only what we need; works under FSDP full-shard.
    """
    @staticmethod
    def forward(ctx, x, weight, u, v, scale, bias=None):
        y = x.matmul(weight.t() + scale * (u @ v))
        ctx.has_bias = bias is not None
        ctx.scale = scale
        ctx.save_for_backward(x, weight, u, v)
        return y if bias is None else y + bias

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, u, v = ctx.saved_tensors
        s = ctx.scale
        # grad_x via effective transpose
        M_T = weight + s * (v.t().matmul(u.t()))
        grad_x = grad_out.matmul(M_T.to(grad_out.dtype))
        # collapse batch dims
        if x.ndim == 2:
            x2, go2 = x, grad_out
        else:
            B = x.numel() // x.shape[-1]
            x2  = x.reshape(B, x.shape[-1])
            go2 = grad_out.reshape(B, grad_out.shape[-1])
        # vanilla dW (out,in)
        grad_weight = go2.t().matmul(x2.to(go2.dtype))
        # NS-processed map to U,V (do NS in stable dtype)
        ns = zeropower_via_newtonschulz5(grad_weight, steps=5).to(v.dtype)  # (out,in)
        grad_u = (v.matmul(ns)).t() * s        # (in,r)
        grad_v = (ns.matmul(u)).t() * s        # (r,out)
        grad_bias = grad_out.sum(dim=tuple(range(grad_out.ndim - 1))) if ctx.has_bias else None
        # grads: (x, weight, u, v, scale, bias); scale is buffer/hparam -> None
        return grad_x, grad_weight, grad_u, grad_v, None, grad_bias

class LoraLinearNS(nn.Module):
    """
    FSDP-friendly LoRA linear:
      - params: weight, bias, lora_A, lora_B
      - buffer: lora_scale (alpha/r), moved/casted by FSDP
    """
    def __init__(self, in_features, out_features, r, bias=True, alpha=1.0, dtype=None, device=None, **_):
        super().__init__()
        assert r > 0
        self.in_features, self.out_features, self.r = in_features, out_features, r
        self.alpha = float(alpha)

        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        self.bias   = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device)) if bias else None

        self.lora_A = nn.Parameter(torch.zeros(in_features, r, dtype=dtype, device=device))  # U
        self.lora_B = nn.Parameter(torch.zeros(r, out_features, dtype=dtype, device=device))  # V

        # scale is a BUFFER so FSDP handles device/dtype & state_dict; no per-forward tensor creation
        self.register_buffer("lora_scale", torch.tensor(self.alpha / float(r), dtype=dtype or torch.float32, device=device))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.normal_(self.lora_A, std=1.0 / max(1, in_features))

    def forward(self, x):
        return LinearFunction.apply(x, self.weight, self.lora_A, self.lora_B, self.lora_scale, self.bias)

# --- Patch PEFT before model construction so VERL uses our class on the ACTOR
try:
    import peft.tuners.lora.layer as loralayer
    loralayer.LoraLinear = LoraLinearNS
    print("[sitecustomize] Patched PEFT LoraLinear -> LoraLinearNS (FSDP-safe; alpha/r; NS backward).")
except Exception as e:
    print("[sitecustomize] WARNING: failed to patch PEFT LoraLinear:", e)
