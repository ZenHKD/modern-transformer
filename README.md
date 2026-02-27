# Modern Transformer

A clean, research-grade PyTorch implementation of a **unified Transformer architecture** that supports all three major paradigms in a single codebase:

- 🧠 **Decoder-Only** — Autoregressive LLMs (e.g., GPT, LLaMA, Gemma)
- 📖 **Encoder-Only** — Bidirectional models (e.g., BERT, ModernBERT)
- 🔀 **Encoder-Decoder** — Seq2Seq models (e.g., T5, BART)

The implementation incorporates state-of-the-art techniques from modern frontier models, making it ideal for educational purposes and as a research baseline.

---

## ✨ Features

| Feature | Description |
|---|---|
| **GQA** | Grouped Query Attention — reduces KV cache size vs. standard MHA |
| **RoPE** | Rotary Positional Embedding with NTK-aware dynamic scaling |
| **RMSNorm** | Pre-normalization for training stability (no mean shift) |
| **QK-Norm** | Per-head query/key normalization to prevent attention entropy collapse |
| **SwiGLU** | Gated activation function used in LLaMA/Gemma FFN layers |
| **Expert-Choice MoE** | Mixture-of-Experts with shared experts and load-balancing aux loss |
| **Sliding Window Attention** | Local attention window for efficient long-context processing |
| **Static KV Cache** | Pre-allocated cache for high-throughput autoregressive inference |
| **DropPath** | Stochastic depth regularization for better generalization |
| **Logit Capping** | Soft-caps attention logits (via `tanh`) to prevent attention spikes |
| **Gradient Checkpointing** | Memory-efficient training by recomputing activations |
| **BF16 Support** | Native bfloat16 for modern accelerators |
| **Weight Tying** | Shares embedding and LM head weights to reduce parameter count |

---

## 📁 Repository Structure

```
modern-transformer/
├── model.py              # Full model implementation (all architectures)
├── configs/
│   ├── bert_1_7b.yaml    # Encoder-Only config (~1.7B params)
│   ├── gemma3_1b.yaml    # Decoder-Only config (~1.29B params)
│   └── llama3_2b.yaml    # Encoder-Decoder config (~2B params)
└── architecture/
    ├── bert_1_7b_summary.txt    # Detailed layer-by-layer summary
    ├── gemma3_1b_summary.txt    # Detailed layer-by-layer summary
    └── llama3_2b_summary.txt    # Detailed layer-by-layer summary
```

---

## 🏗️ Architecture Overview

All three architecture modes are built from a shared set of composable building blocks:

```
TransformerModel
├── embed_tokens (nn.Embedding)
├── TransformerEncoder   [encoder_only | encoder_decoder]
│   └── EncoderBlock × N
│       ├── AttentionBlock  (Self-Attention, bidirectional)
│       └── ExpertChoiceMoE (SwiGLU experts + shared experts)
├── TransformerDecoder   [decoder_only | encoder_decoder]
│   └── DecoderBlock × N
│       ├── AttentionBlock  (Masked Self-Attention + Static KV Cache)
│       ├── AttentionBlock  (Cross-Attention) [encoder_decoder only]
│       └── ExpertChoiceMoE (SwiGLU experts + shared experts)
└── head (Linear LM head or classification head)
```

### Key Components

- **`BlockConfig`** — Per-block hyperparameters (layers, heads, FFN dim, MoE settings, etc.)
- **`ModelConfig`** — Global model settings (architecture type, RoPE, vocab, weight tying, etc.)
- **`AttentionBlock`** — Unified attention supporting self-attention and cross-attention, with GQA + RoPE + QK-Norm + Sliding Window
- **`ExpertChoiceMoE`** — Token-choice MoE with top-k routing, capacity limiting, and always-on shared experts
- **`StaticCache`** — Pre-allocated KV cache addressed by `(batch, layer, head, position)`
- **`RotaryPositionalEmbedding`** — Cached cosine/sine tables with optional NTK scaling for long-context generalization

---

## ⚙️ Pre-built Configurations

### Gemma 3 1B — Decoder-Only

```yaml
# configs/gemma3_1b.yaml
architecture: "decoder_only"
vocab_size: 256000
decoder:
  d_model: 1152
  n_layers: 26
  n_heads: 4
  n_kv_heads: 4
  d_ff: 11264
  max_seq_len: 131072   # 128K context
rope_scaling: "ntk"     # NTK-aware dynamic RoPE
```
~1.29B parameters (dense), 128K context window.

---

### ModernBERT 1.7B — Encoder-Only

```yaml
# configs/bert_1_7b.yaml
architecture: "encoder_only"
vocab_size: 128256
encoder:
  d_model: 2304
  n_layers: 20
  n_heads: 16
  n_kv_heads: 4          # GQA with 4 KV heads
  d_ff: 8192
  max_seq_len: 4096
qk_norm: true
```
~1.7B parameters (dense), bidirectional encoder with GQA.

---

### LLaMA 3 2B — Encoder-Decoder

```yaml
# configs/llama3_2b.yaml
architecture: "encoder_decoder"
vocab_size: 128256
encoder:
  d_model: 2048
  n_layers: 12
  n_heads: 16 / n_kv_heads: 4
decoder:
  d_model: 2048
  n_layers: 12
  n_heads: 16 / n_kv_heads: 4
rope_theta: 500000.0
```
~2B parameters (dense), seq2seq architecture.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch pyyaml
```

### 2. Run a Forward Pass

```bash
# Decoder-Only (LLM)
python model.py configs/gemma3_1b.yaml

# Encoder-Only (BERT-like)
python model.py configs/bert_1_7b.yaml

# Encoder-Decoder (Seq2Seq)
python model.py configs/llama3_2b.yaml
```

### 3. Use in Your Own Code

```python
from model import TransformerModel

# Load from YAML config
model = TransformerModel.from_yaml("configs/gemma3_1b.yaml")
print(f"Parameters: {model.get_num_params() / 1e6:.1f}M")

# Decoder-Only forward pass
import torch
input_ids = torch.randint(0, 256000, (1, 512))
logits, aux_loss = model(input_ids)
print(logits.shape)  # (1, 512, 256000)
```

```python
# Encoder-Decoder forward pass
model = TransformerModel.from_yaml("configs/llama3_2b.yaml")
encoder_input = torch.randint(0, 128256, (1, 256))
decoder_input = torch.randint(0, 128256, (1, 64))
logits, aux_loss = model(input_ids=encoder_input, decoder_input_ids=decoder_input)
```

### 4. Export Model Summary

```python
from model import export_model_summary_to_file

export_model_summary_to_file(
    config_path="configs/gemma3_1b.yaml",
    batch_size=1,
    seq_len=8192,
    output_filename="gemma3_summary.txt"
)
```

This exports a detailed report including parameter counts per layer and memory estimates (weights, KV cache, training with AdamW).

---

## 🔧 Custom Configuration

Create your own YAML config with the following schema:

```yaml
architecture: "decoder_only"   # or "encoder_only" / "encoder_decoder"
vocab_size: 32000
weight_tying: true
use_bf16: true
use_gradient_checkpointing: true

# RoPE
apply_rope: true
rope_theta: 500000.0
rope_fraction: 1.0
rope_scaling: null             # null, "ntk"
rope_in_cross_attn: false

# Attention
qk_norm: true
logit_cap: 30.0

# KV Cache (inference)
cache_mode: "static"
num_sink_tokens: 4

decoder:                       # use "encoder:" for encoder-only
  d_model: 1024
  n_layers: 16
  n_heads: 8
  n_kv_heads: 2               # GQA: < n_heads for KV compression
  d_ff: 4096
  max_seq_len: 8192
  sliding_window: null        # set int for local attention
  dropout: 0.0
  drop_path: 0.0
  use_bias: false
  init_std: 0.02
  num_experts: 1              # 1 = dense; > 1 enables MoE
  top_k: 1
  num_shared_experts: 0
  capacity_factor: 1.2
  aux_loss_coef: 0.01
  router_z_loss_coef: 0.001
  noise_std: 0.1
```

### MoE Configuration

To enable Mixture-of-Experts, set `num_experts > 1`:

```yaml
num_experts: 8      # number of routed experts per layer
top_k: 2            # tokens select top-k experts
num_shared_experts: 2  # experts that see all tokens (always active)
capacity_factor: 1.2   # expert capacity = (tokens * capacity_factor) / num_experts
aux_loss_coef: 0.01    # load balancing loss weight
router_z_loss_coef: 0.001
```

The `aux_loss` returned by the model's forward pass should be added to the cross-entropy loss during training:
```python
logits, aux_loss = model(input_ids)
loss = cross_entropy_loss + aux_loss
```

---

## 📊 Memory Estimates

Memory estimates are generated automatically by `export_model_summary_to_file`. Below are rough estimates for the included configurations at BF16:

| Model | Params | Weights | Inference (8K ctx) | Training (AdamW) |
|---|---|---|---|---|
| Gemma 3 1B | ~1.29B | ~2.4 GB | ~3.1 GB | ~9.7 GB |
| ModernBERT 1.7B | ~1.7B | ~3.2 GB | ~3.8 GB | ~12.8 GB |
| LLaMA 3 2B | ~2B | ~3.7 GB | ~4.5 GB | ~14.9 GB |

> **Note**: Training memory is a lower bound. Real activation memory with gradient checkpointing can be 2–4× higher depending on batch size and sequence length.

---

## 🤝 Contributing

This project is primarily educational. Issues and PRs are welcome for bug fixes, new config files, or additional features (e.g., speculative decoding, FlashAttention-2 kernel integration).

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
