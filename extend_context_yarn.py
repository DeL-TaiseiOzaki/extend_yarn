"""
YaRNを使用してコンテキストウィンドウを拡張し、Hugging Faceに直接アップロードするスクリプト
対象モデル: ft-llm-team-mkj/baseline-model-instruct (4K → 8K, 16K, 32K)
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import transformers
import torch
from packaging import version

# ベースモデルの設定
BASE_MODEL = "ft-llm-team-mkj/baseline-model-instruct"
ORIGINAL_MAX_POS = 4096
ROPE_THETA = 500000.0

# Hugging Faceのユーザー名/Organization
HF_USERNAME = "DeL-TaiseiOzaki"

# 拡張設定: (新しいコンテキスト長, factor, リポジトリ名)
EXTENSIONS = [
    (8192, 2.0, "baseline-model-instruct-8k-yarn"),
    (16384, 4.0, "baseline-model-instruct-16k-yarn"),
    (32768, 8.0, "baseline-model-instruct-32k-yarn"),
]


def get_transformers_version():
    """transformersのバージョンを取得"""
    return version.parse(transformers.__version__)


def create_yarn_config(base_config, new_max_pos: int, factor: float):
    """YaRN設定を適用したconfigを作成"""
    tf_version = get_transformers_version()
    
    # max_position_embeddingsを更新
    base_config.max_position_embeddings = new_max_pos
    
    if tf_version >= version.parse("4.45.0"):
        # 最新形式: rope_parameters
        print(f"  Using rope_parameters format (transformers {tf_version})")
        base_config.rope_parameters = {
            "rope_type": "yarn",
            "rope_theta": ROPE_THETA,
            "factor": factor,
            "original_max_position_embeddings": ORIGINAL_MAX_POS,
            "beta_fast": 32,
            "beta_slow": 1,
        }
        if hasattr(base_config, 'rope_scaling'):
            base_config.rope_scaling = None
            
    elif tf_version >= version.parse("4.36.0"):
        # 中間形式: rope_scaling
        print(f"  Using rope_scaling format (transformers {tf_version})")
        base_config.rope_scaling = {
            "type": "yarn",
            "factor": factor,
            "original_max_position_embeddings": ORIGINAL_MAX_POS,
            "beta_fast": 32,
            "beta_slow": 1,
        }
    else:
        raise RuntimeError(
            f"transformers {tf_version} does not support YaRN. "
            f"Please upgrade: pip install -U transformers"
        )
    
    return base_config


def extend_and_upload_model(new_max_pos: int, factor: float, repo_name: str):
    """モデルを拡張してHugging Faceにアップロード"""
    
    repo_id = f"{HF_USERNAME}/{repo_name}"
    
    print(f"\n{'='*60}")
    print(f"Creating {new_max_pos // 1024}K context model (factor={factor})")
    print(f"Repository: {repo_id}")
    print('='*60)
    
    # 1. Configをロードして修正
    print("Loading and modifying config...")
    config = AutoConfig.from_pretrained(BASE_MODEL)
    config = create_yarn_config(config, new_max_pos, factor)
    
    # 2. 修正したconfigでモデルをロード
    print("Loading model with new config...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 3. Tokenizerをロード
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    # 4. Hugging Faceにアップロード
    print(f"Uploading to {repo_id}...")
    
    # モデルカードの内容
    model_card = f"""---
base_model: {BASE_MODEL}
tags:
- yarn
- long-context
- llama
license: other
---

# {repo_name}

This model is [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}) with YaRN (Yet another RoPE extensioN) applied to extend the context window.

## Model Details

- **Base Model**: {BASE_MODEL}
- **Original Context Length**: {ORIGINAL_MAX_POS} tokens
- **Extended Context Length**: {new_max_pos} tokens
- **Extension Method**: YaRN
- **Scaling Factor**: {factor}

## YaRN Configuration

```json
{{
  "rope_type": "yarn",
  "factor": {factor},
  "original_max_position_embeddings": {ORIGINAL_MAX_POS},
  "beta_fast": 32,
  "beta_slow": 1
}}
```

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "{repo_id}",
    torch_dtype="bfloat16",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```

## Note

This model has YaRN applied but has **not been fine-tuned** on long-context data yet.
For optimal performance on long sequences, fine-tuning on long documents is recommended.

## References

- [YaRN Paper (ICLR 2024)](https://openreview.net/forum?id=wHBfxhZu1u)
- [YaRN GitHub](https://github.com/jquesnelle/yarn)
"""
    
    # アップロード実行
    model.push_to_hub(
        repo_id,
        commit_message=f"Upload YaRN extended model ({new_max_pos // 1024}K context)",
        private=False,  # 公開リポジトリ。非公開にしたい場合はTrue
    )
    
    tokenizer.push_to_hub(
        repo_id,
        commit_message=f"Upload tokenizer for YaRN extended model",
    )
    
    # README.mdをアップロード
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Add model card",
    )
    
    print(f"✓ Uploaded: https://huggingface.co/{repo_id}")
    
    # メモリ解放
    del model
    torch.cuda.empty_cache()


def main():
    tf_version = get_transformers_version()
    
    print("="*60)
    print("YaRN Context Extension & Upload Script")
    print("="*60)
    print(f"transformers version: {tf_version}")
    print(f"Base model: {BASE_MODEL}")
    print(f"Original context: {ORIGINAL_MAX_POS}")
    print(f"Target contexts: {[f'{ext[0]//1024}K' for ext in EXTENSIONS]}")
    print(f"Upload to: {HF_USERNAME}")
    
    # バージョンチェック
    if tf_version < version.parse("4.36.0"):
        print(f"\n❌ Error: transformers {tf_version} does not support YaRN.")
        print("Please upgrade: pip install -U transformers")
        return
    
    for new_max_pos, factor, repo_name in EXTENSIONS:
        extend_and_upload_model(new_max_pos, factor, repo_name)
    
    print("\n" + "="*60)
    print("✓ All models uploaded successfully!")
    print("="*60)
    print(f"\nUploaded repositories:")
    for _, _, repo_name in EXTENSIONS:
        print(f"  - https://huggingface.co/{HF_USERNAME}/{repo_name}")


if __name__ == "__main__":
    main()