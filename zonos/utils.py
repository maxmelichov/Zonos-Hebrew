import torch
import torch.nn as nn
import torch.nn.functional as F


def find_multiple(n: int, k: int) -> int:
    if k == 0 or n % k == 0:
        return n
    return n + k - (n % k)


def pad_weight_(w: nn.Embedding | nn.Linear, multiple: int):
    """Pad the weight of an embedding or linear layer to a multiple of `multiple`."""
    if isinstance(w, nn.Embedding):
        # Pad input dim
        if w.weight.shape[1] % multiple == 0:
            return
        w.weight.data = F.pad(w.weight.data, (0, 0, 0, w.weight.shape[1] % multiple))
        w.num_embeddings, w.embedding_dim = w.weight.shape
    elif isinstance(w, nn.Linear):
        # Pad output dim
        if w.weight.shape[0] % multiple == 0:
            return
        w.weight.data = F.pad(w.weight.data, (0, 0, 0, w.weight.shape[0] % multiple))
        w.out_features, w.in_features = w.weight.shape
    else:
        raise ValueError(f"Unsupported weight type: {type(w)}")


def pad_weight_train_(w: nn.Embedding | nn.Linear, multiple: int):
    """Training-time padding behavior.

    Goal: keep vocabulary/head sizes stable and aligned with inference's current expectation (1026),
    avoiding incremental growth. We therefore:
      - Do nothing for embeddings (they are already 1026 in this codebase)
      - For linear heads: if out_features is 1025, pad exactly one row to reach 1026; otherwise no-op
    """
    if isinstance(w, nn.Embedding):
        return
    if isinstance(w, nn.Linear):
        current_out = w.weight.shape[0]
        if current_out == 1025:
            # Pad exactly one row to match 1026 expected size
            w.weight.data = F.pad(w.weight.data, (0, 0, 0, 1))
            w.out_features, w.in_features = w.weight.shape
        return
    raise ValueError(f"Unsupported weight type: {type(w)}")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(torch.cuda.current_device())
    # MPS breaks for whatever reason. Uncomment when it's working.
    # if torch.mps.is_available():
    #     return torch.device("mps")
    return torch.device("cpu")


DEFAULT_DEVICE = get_device()
