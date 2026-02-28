# Modern Transformer

A single-file PyTorch implementation of a modern Transformer, supporting all three architectures:

- **Decoder-Only** — LLMs (GPT, LLaMA, Gemma style)
- **Encoder-Only** — BERT-like models
- **Encoder-Decoder** — Seq2Seq (T5, BART style)

---

## What's Inside

Everything lives in `model.py` (~1000 lines).

**Attention**
- Grouped Query Attention (GQA)
- Rotary Positional Embedding (RoPE) with NTK scaling
- QK-Norm, Soft Logit Capping
- Sliding Window Attention
- Static KV Cache for inference
- **Gated DeltaNet** — linear O(N) attention via the Delta Rule (Qwen3-Next style)
- **Hybrid layout** — interleave DeltaNet and softmax layers (3:1 ratio)

**FFN / MoE**
- SwiGLU activation
- **Token-Choice MoE** — each token picks top-k experts (Mixtral/Qwen3 style)
- Expert-Choice MoE (original, kept for reference)
- Shared experts (always active)

**Training**
- Pre-Norm with RMSNorm
- DropPath (Stochastic Depth)
- Gradient Checkpointing
- BF16 support, Weight Tying

---

## Configs

| Config | Architecture | Attention | ~Params |
|---|---|---|---|
| `gemma3_1b.yaml` | Decoder-Only | Softmax | 1.29B |
| `bert_1_7b.yaml` | Encoder-Only | Softmax | 1.7B |
| `llama3_2b.yaml` | Encoder-Decoder | Softmax | 2B |
| `qwen3_mini.yaml` | Decoder-Only | **Hybrid** (DeltaNet) | small |

---

## Quick Start

```bash
pip install torch pyyaml
```

```bash
# Run any config
python model.py configs/gemma3_1b.yaml
python model.py configs/qwen3_mini.yaml   # hybrid linear attention
```

```python
from model import TransformerModel
import torch

model = TransformerModel.from_yaml("configs/gemma3_1b.yaml")
print(f"{model.get_num_params() / 1e6:.1f}M params")

input_ids = torch.randint(0, 256000, (1, 512))
logits, aux_loss = model(input_ids)
```

---

## Custom Config

```yaml
architecture: "decoder_only"   # "encoder_only" | "encoder_decoder"
vocab_size: 32000
weight_tying: true

decoder:
  d_model: 1024
  n_layers: 16
  n_heads: 8
  n_kv_heads: 2          # GQA: fewer KV heads = smaller cache
  d_ff: 4096

  # Attention: "softmax" | "linear" | "hybrid"
  attention_type: "hybrid"
  hybrid_ratio: 3        # 3 DeltaNet layers per 1 softmax layer

  # MoE: "token_choice" | "expert_choice"
  moe_routing: "token_choice"
  num_experts: 8
  top_k: 2
  num_shared_experts: 1
```