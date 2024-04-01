import functools

from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


def get_llama_wrapper():
    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            MixtralDecoderLayer,
        },
    )

    return llama_auto_wrap_policy
