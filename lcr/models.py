"""LCR (local contact residual) and the baselines it is compared against."""
import torch
import torch.nn as nn
import torch.nn.functional as Fn

N_GEOM_START = 13          # feats[..., 13:] are the two SDF patches (27 + 27)
WRENCH_SCALE = torch.tensor([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])


def mlp(i, h, o, layers=2):
    mods, d = [], i
    for _ in range(layers - 1):
        mods += [nn.Linear(d, h), nn.SiLU()]
        d = h
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


class LCR(nn.Module):
    """Per-contact residual forces with physical shaping, summed into a 6-D wrench."""

    def __init__(self, n_feat=67, n_glob=12, h=64, use_geom=True, attention=True, mu_max=1.0):
        super().__init__()
        self.use_geom, self.mu_max = use_geom, mu_max
        self.enc = mlp(n_feat, 128, h, 3)
        self.glob = mlp(n_glob, 64, h, 2)
        self.attn = nn.MultiheadAttention(h, 4, batch_first=True) if attention else None
        self.norm = nn.LayerNorm(h)
        self.head = mlp(h, 64, 4, 2)
        nn.init.zeros_(self.head[-1].weight)
        with torch.no_grad():
            self.head[-1].bias.copy_(torch.tensor([0.0, -6.0, 0.0, 0.0]))

    def forward(self, b):
        x = b["feats"]
        if not self.use_geom:
            x = torch.cat([x[..., :N_GEOM_START], torch.zeros_like(x[..., N_GEOM_START:])], -1)
        mask = b["mask"]
        h = self.enc(x)
        g = self.glob(b["glob"]).unsqueeze(1)
        if self.attn is not None:
            tok = torch.cat([g, h], 1)
            pad = torch.cat([torch.zeros_like(mask[:, :1]), 1 - mask], 1).bool()
            a, _ = self.attn(tok, tok, tok, key_padding_mask=pad, need_weights=False)
            h = self.norm(h + a[:, 1:])
        else:
            h = self.norm(h + g)
        o = self.head(h)
        lam = b["lam"]
        f_n = lam[..., 0].clamp(min=0) * torch.exp(o[..., 0].clamp(-3, 3)) + 10.0 * Fn.softplus(o[..., 1])
        f_t = lam[..., 1:] + 10.0 * o[..., 2:4]
        nt = f_t.norm(dim=-1, keepdim=True) + 1e-6
        f_t = f_t * torch.clamp(self.mu_max * f_n.unsqueeze(-1) / nt, max=1.0)
        f = torch.cat([f_n.unsqueeze(-1), f_t], -1)
        df = (f - lam) * mask.unsqueeze(-1)                       # contact frame (n, t1, t2)
        df_w = torch.einsum("bkij,bki->bkj", b["R"], df)          # rows of R are n, t1, t2
        F = df_w.sum(1)
        T = torch.cross(b["r"], df_w, dim=-1).sum(1)
        return torch.cat([F, T], -1)


class GlobalResidual(nn.Module):
    """Paper-style residual: 3-layer MLP (hidden 128) from pose/velocity/command to a 6-D wrench."""

    def __init__(self, n_in=21):
        super().__init__()
        self.net = mlp(n_in, 128, 6, 3)
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, b):
        return self.net(torch.cat([b["glob"], b["pose"]], -1)) * WRENCH_SCALE


class BlackBox(nn.Module):
    """No simulator: predicts the next pose increment and the F/T reading directly."""

    def __init__(self, n_in=21):
        super().__init__()
        self.net = mlp(n_in, 256, 12, 3)

    def forward(self, b):
        return self.net(torch.cat([b["glob"], b["pose"]], -1))
