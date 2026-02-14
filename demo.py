from importlib.metadata import PackageNotFoundError, version

import torch
from packaging.version import Version


def build_inputs(vocab_size, seq_len, device):
    input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def is_hub_compatible():
    try:
        hub_version = Version(version("huggingface_hub"))
    except PackageNotFoundError:
        return False
    return Version("0.34.0") <= hub_version < Version("1.0.0")


def main():
    if not is_hub_compatible():
        print("huggingface-hub version is incompatible, install a version in [0.34.0, 1.0.0).")
        return
    from scripts.model.configuration_minicpm import MiniCPM3Config
    from scripts.model.modeling_minicpm_solo_vst_lge5 import MiniCPM3ForCausalLM

    config = MiniCPM3Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    config._attn_implementation = "sdpa"
    model = MiniCPM3ForCausalLM(config)
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    input_ids, attention_mask = build_inputs(config.vocab_size, 16, device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    print(outputs.logits.shape)


if __name__ == "__main__":
    main()
