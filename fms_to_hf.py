import fire
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from torch.distributed._shard.checkpoint import FileSystemReader, load_state_dict


def main(load_path, save_path, tokenizer_name_or_path):
    print("Initializing model...")
    config_data = {
        "d_model": 2048,
        "d_intermediate": 0,
        "n_layer": 48,
        "vocab_size": 128256,
        "ssm_cfg": {"layer": "Mamba2"},
        "attn_layer_idx": [8, 16, 24, 32, 40],
        "attn_cfg": {
            "causal": True,
            "d_conv": 4,
            "head_dim": 128,
            "num_heads": 24,
            "out_proj_bias": False,
            "qkv_proj_bias": False,
            "rotary_emb_dim": 64,
        },
        "rms_norm": True,
        "residual_in_fp32": True,
        "fused_add_norm": True,
        "pad_vocab_size_multiple": 16,
        "tie_embeddings": True,
    }
    mamba_config = MambaConfig(**config_data)
    model = MambaLMHeadModel(mamba_config)

    print(f"Reading state dict from {load_path}")
    state_dict = {"model_state": model.state_dict()}
    load_state_dict(
        state_dict=state_dict, storage_reader=FileSystemReader(load_path), no_dist=True
    )

    print("Loading state dict into the model...")
    model.load_state_dict(state_dict["model_state"])

    print("Saving model to HF-compatible format...")
    model.save_pretrained(save_path)

    print("Copying tokenizer...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    tokenizer.save_pretrained(save_path)

    print(f"Model saving at {save_path}")


if __name__ == "__main__":
    fire.Fire(main)
