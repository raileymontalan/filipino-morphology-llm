"""
A collection of positional encoding modules.
"""

import math

import torch


class LearnedPosEncoding(torch.nn.Module):
    """Basic learned positional encoding."""

    def __init__(self, hidden_dim, context_window):
        """Initialize learned positional encoding."""
        super().__init__()
        self.pe = torch.nn.Embedding(num_embeddings=context_window, embedding_dim=hidden_dim)

    def forward(self, x):
        """Return input tensor with positional encoding added."""
        if len(x.shape) >= 2:
            return x + (self.pe(torch.arange(x.size(1), device=x.device)).unsqueeze(0))
        else:
            return x + self.pe(torch.arange(x.size(1), device=x.device))


class IdentityEncoding(torch.nn.Module):
    """Identity positional encoding (no-op)."""

    def __init__(self):
        """Initialize identity encoding."""
        super().__init__()

    def forward(self, x):
        """Return input tensor unchanged."""
        return x


class SinCosPosEncoding(torch.nn.Module):
    """SinCos positional encoding as in Vaswani et al."""

    def __init__(self, hidden_dim, context_window):
        """Initialize SinCos positional encoding."""
        super().__init__()
        pe = torch.zeros(context_window, hidden_dim)
        position = torch.arange(0, context_window, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # pe has shape (1, S, H)

        self.pe = torch.nn.Parameter(pe)  # hack for distributed data parallel
        self.pe.requires_grad = False

    def forward(self, x):
        """Return input tensor with SinCos positional encoding added."""
        # x of shape (B, S, H)
        return x + self.pe[:, : x.size(1)]


POS_ENCODING_DICT = {
    "learned": lambda dim, size, **_: LearnedPosEncoding(hidden_dim=dim, context_window=size),
    "rope": lambda **_: IdentityEncoding(),
    "none": lambda **_: IdentityEncoding(),
    "sincos": lambda dim, size, **_: SinCosPosEncoding(hidden_dim=dim, context_window=size),
}


def build_positional_encodings(model_cfg):
    """Build positional encoding module from config."""
    return POS_ENCODING_DICT[model_cfg["positional_encoding_type"]](
        dim=model_cfg["hidden_dim"], size=model_cfg["context_window"]
    )
