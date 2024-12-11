import torch
import torch.nn as nn
import numpy as np
import utils


class VisionTransformer(nn.Module):
    def __init__(self, config, n_classes, name):
        super().__init__()

        # Extract model configuration
        config = config[name]
        d_model = config["d_model"]
        n_layers = config["n_layers"]
        n_heads = config["n_heads"]
        img_size = config["img_size"]
        patch_size = config["patch_size"]
        n_channels = config["n_channels"]

        assert (
            img_size % patch_size
        ) == 0, "Image dimensions must be divisible by the patch size."
        assert (
            d_model % n_heads == 0
        ), "Embedding dimension must be divisible by the number of heads."

        self.d_model = d_model  # Dimension of the model
        self.n_classes = n_classes  # Number of classes
        self.img_size = img_size  # Image size
        self.patch_size = patch_size  # Patch size
        self.n_channels = n_channels  # Number of channels
        self.n_heads = n_heads  # Number of heads

        self.n_patches = (self.img_size**2) // (self.patch_size**2)
        self.max_len = self.n_patches + 1

        self.patch_embedding = PatchEmbedding(
            self.img_size, self.patch_size, self.n_channels, self.d_model
        )
        self.positional_encoding = PositionalEncoding(self.d_model, self.max_len)
        self.transformer_encoder = nn.Sequential(
            *[TransformerEncoder(self.d_model, self.n_heads) for _ in range(n_layers)]
        )

        # Classification MLP
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.n_classes), nn.Softmax(dim=-1)
        )

    def forward(self, images):
        x = self.patch_embedding(images)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x[:, 0, :]
        x = self.classifier(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, d_model):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans, d_model, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # (B, E, P, P)
        x = x.flatten(2)  # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        pe = torch.zeros(max_len, embed_dim)

        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        tokens_batch = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((tokens_batch, x), dim=1)
        return x + self.pe[:, : x.size(1)]


class AttentionHead(nn.Module):
    def __init__(self, d_model, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(d_model, head_size)
        self.key = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        attention = torch.einsum("bqe,bte->bqt", query, key) / self.head_size**0.5
        attention = torch.softmax(attention, dim=-1)
        attention = torch.einsum("bqt,bte->bqe", attention, value)

        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.head_size = d_model // n_heads
        self.W_o = nn.Linear(d_model, d_model)
        self.heads = nn.ModuleList(
            [AttentionHead(d_model, self.head_size) for _ in range(n_heads)]
        )

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.W_o(out)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, r_mlp=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Sublayer 1 - Normalization
        self.norm1 = nn.LayerNorm(d_model)

        # Sublayer 2 - Multi-head attention
        self.attn = MultiHeadAttention(d_model, n_heads)

        # Sublayer 3 - Normalization
        self.norm2 = nn.LayerNorm(d_model)

        # Sublayer 4 - Multi-layer perceptron
        self.mlp = nn.Sequential(
            nn.Linear(d_model, r_mlp * d_model),
            nn.GELU(),
            nn.Linear(r_mlp * d_model, d_model),
        )

    def forward(self, x):
        # Sublayer 1 - Normalization
        out = self.norm1(x)

        # Sublayer 2 - Multi-head attention
        out = self.attn(out) + x

        # Sublayer 3 - Normalization
        out = self.norm2(out)

        # Sublayer 4 - Multi-layer perceptron
        out = self.mlp(out) + out

        return out
