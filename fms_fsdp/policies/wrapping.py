import functools

from torch.distributed.fsdp.wrap import (
    _or_policy,
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
)


def get_wrapper(cfg, block):
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={block},
    )

    lambda_policy = functools.partial(
        lambda_auto_wrap_policy,
        lambda_fn=lambda module: getattr(module, "out_features", -1) == cfg.vocab_size,
    )

    auto_wrap_policy = functools.partial(
        _or_policy, policies=[transformer_wrap_policy, lambda_policy]
    )
    return auto_wrap_policy
