import functools

from fms.modules.embedding import WordEmbedding
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


def get_wrapper(block):
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            block, WordEmbedding
        },
    )

    return auto_wrap_policy
