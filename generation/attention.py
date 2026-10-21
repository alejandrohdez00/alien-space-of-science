"""Keep mathematically redundant attention key biases at zero during training."""
import torch


@torch.no_grad()
def zero_key_bias(raw, gradients=False):
    # Key bias adds the same constant to every attention score in a row and
    # cancels in exact softmax. Suppress its finite-precision optimizer drift.
    for layer in raw.set_transformer.layers:
        bias = layer.self_attn.in_proj_bias
        value = bias.grad if gradients else bias
        if value is not None:
            value.chunk(3)[1].zero_()
