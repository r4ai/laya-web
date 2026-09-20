"""Export-only PyTorch port of laya-mlx (Apache-2.0; see NOTICE).

Single-question, unpadded inference. Embeddings are gathered on the CPU so a
large multilingual vocabulary never becomes a WebGPU storage buffer.
"""

import torch
from torch import nn
from torch.nn import functional as F


def norm(dims, eps=1e-5, bias=True):
    return nn.LayerNorm(dims, eps=eps, bias=bias)


class Attention(nn.Module):
    def __init__(
        self, dims, heads, bias, *, encoder=False, base=10000, max_length=1024
    ):
        super().__init__()
        self.heads, self.width = heads, dims // heads
        self.encoder = encoder
        if encoder:
            self.Wqkv = nn.Linear(dims, dims * 3, bias=bias)
            self.Wo = nn.Linear(dims, dims, bias=bias)
            freq = 1 / base ** (torch.arange(0, self.width, 2).float() / self.width)
            angles = torch.arange(max_length).float()[:, None] * freq[None, :]
            angles = torch.cat([angles, angles], dim=-1)[None, None]
            self.register_buffer("cos", angles.cos(), persistent=False)
            self.register_buffer("sin", angles.sin(), persistent=False)
        else:
            self.in_proj = nn.Linear(dims, dims * 3)
            self.out_proj = nn.Linear(dims, dims)

    def rotate(self, x):
        a, b = x.chunk(2, dim=-1)
        n = x.shape[2]
        return x * self.cos[:, :, :n] + torch.cat([-b, a], dim=-1) * self.sin[:, :, :n]

    def forward(self, x, mask=None):
        b, n, d = x.shape
        projection = self.Wqkv if self.encoder else self.in_proj
        qkv = projection(x).reshape(b, n, 3, self.heads, self.width)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        if self.encoder:
            q, k = self.rotate(q), self.rotate(k)
        weights = (q @ k.transpose(-1, -2)) * self.width**-0.5
        if mask is not None:
            weights = weights + mask
        out = (weights.softmax(-1) @ v).transpose(1, 2).reshape(b, n, d)
        return (self.Wo if self.encoder else self.out_proj)(out)


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        d, h = c["hidden_size"], c["intermediate_size"]
        self.Wi = nn.Linear(d, 2 * h, bias=c.get("mlp_bias", False))
        self.Wo = nn.Linear(h, d, bias=c.get("mlp_bias", False))

    def forward(self, x):
        value, gate = self.Wi(x).chunk(2, dim=-1)
        return self.Wo(F.gelu(value) * gate)


class EncoderLayer(nn.Module):
    def __init__(self, c, i, max_length):
        super().__init__()
        d = c["hidden_size"]
        kind = c.get(
            "layer_types",
            [
                "full_attention"
                if j % c.get("global_attn_every_n_layers", 3) == 0
                else "sliding_attention"
                for j in range(c["num_hidden_layers"])
            ],
        )[i]
        self.local = kind == "sliding_attention"
        fallback = (
            c.get("local_rope_theta", 10000)
            if self.local
            else c.get("global_rope_theta", 160000)
        )
        params = c.get("rope_parameters", {}).get(kind, {})
        if params.get("rope_type", "default") != "default":
            raise ValueError("Only unscaled RoPE is supported")
        self.attn_norm = (
            nn.Identity()
            if i == 0
            else norm(d, c.get("norm_eps", 1e-5), c.get("norm_bias", False))
        )
        self.attn = Attention(
            d,
            c["num_attention_heads"],
            c.get("attention_bias", False),
            encoder=True,
            base=params.get("rope_theta", fallback),
            max_length=max_length,
        )
        self.mlp_norm = norm(d, c.get("norm_eps", 1e-5), c.get("norm_bias", False))
        self.mlp = MLP(c)

    def forward(self, x, mask):
        x = x + self.attn(self.attn_norm(x), mask if self.local else None)
        return x + self.mlp(self.mlp_norm(x))


class Encoder(nn.Module):
    def __init__(self, c, max_length):
        super().__init__()
        if (
            c["model_type"] != "modernbert"
            or c.get("hidden_activation", "gelu") != "gelu"
        ):
            raise ValueError("Expected ModernBERT with GELU")
        d = c["hidden_size"]
        self.embeddings = nn.ModuleDict(
            {"norm": norm(d, c.get("norm_eps", 1e-5), c.get("norm_bias", False))}
        )
        self.layers = nn.ModuleList(
            [EncoderLayer(c, i, max_length) for i in range(c["num_hidden_layers"])]
        )
        self.final_norm = norm(d, c.get("norm_eps", 1e-5), c.get("norm_bias", False))
        pos = torch.arange(max_length)
        local = (pos[:, None] - pos[None, :]).abs() <= c.get(
            "local_attention", 128
        ) // 2
        self.register_buffer(
            "mask", torch.where(local, 0.0, -1e9)[None, None], persistent=False
        )

    def forward(self, embeddings):
        x = self.embeddings["norm"](embeddings)
        n = x.shape[1]
        for layer in self.layers:
            x = layer(x, self.mask[:, :, :n, :n])
        return self.final_norm(x)


class HeadLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_attn = Attention(d, max(1, d // 64), True)
        self.norm1, self.norm2 = norm(d), norm(d)
        self.linear1, self.linear2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        x = x + self.self_attn(self.norm1(x))
        return x + self.linear2(F.relu(self.linear1(self.norm2(x))))


class DecisionModel(nn.Module):
    def __init__(self, c, agent):
        super().__init__()
        d = c["hidden_size"]
        self.encoder = Encoder(c, agent["max_len"])
        self.head = nn.ModuleDict(
            {
                "layers": nn.ModuleList(
                    [HeadLayer(d) for _ in range(agent["head_layers"])]
                )
            }
        )
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(
            norm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)
        )
        self.act_head = nn.Sequential(
            nn.Linear(d + 4, 256),
            nn.GELU(),
            nn.Linear(256, len(agent.get("act_costs", {})) + 1),
        )
        self.register_buffer("temperature", torch.ones(3))

    def forward(self, embeddings, marker_pos, marker_mask, qtype):
        h = self.encoder(embeddings) + self.type_emb(qtype)[:, None]
        for layer in self.head["layers"]:
            h = layer(h)
        logits = self.scorer(h[:, marker_pos]).squeeze(-1)
        logits = torch.where(marker_mask[None], logits, -1e4)
        p = logits.softmax(-1)
        k = marker_mask.sum().float().clamp(min=2)
        entropy = -(p * p.clamp(min=1e-9).log()).sum(-1) / k.log()
        top = p.topk(2, dim=-1).values
        features = torch.stack(
            [top[:, 0], top[:, 0] - top[:, 1], entropy, (k / 255).expand_as(entropy)],
            dim=-1,
        )
        return logits, self.act_head(torch.cat([h[:, 0], features], dim=-1))


def load_weights(model, weights):
    """Strictly accept original or MLX parameter names; embeddings are separate."""
    names = {}
    for key, value in weights.items():
        key = key.replace(".in_proj_weight", ".in_proj.weight").replace(
            ".in_proj_bias", ".in_proj.bias"
        )
        key = key.replace("scorer.layers.", "scorer.").replace(
            "act_head.layers.", "act_head."
        )
        names[key] = value.float()
    embeddings = names.pop("encoder.embeddings.tok_embeddings.weight")
    model.load_state_dict(names, strict=True)
    return embeddings
