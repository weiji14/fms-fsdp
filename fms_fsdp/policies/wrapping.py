import functools

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn import Embedding


def get_wrapper(block):
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            block, Embedding
        },
    )

    return auto_wrap_policy
