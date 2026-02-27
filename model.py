import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Any
import os
import sys
import yaml

@dataclass
class BlockConfig:
    # Layer
    d_model: int = 2048
    n_layers: int = 12
    n_heads: int = 16
    n_kv_heads: int = 4
    d_ff: int = 8192

    max_seq_len: int = 4096
    sliding_window: Optional[int] = 4096

    # Regularization
    dropout: float = 0.1
    drop_path: float = 0.0
    
    # Initialization
    init_std: float = 0.02
    use_bias: bool = False

    # MoE params
    num_experts: int = 8
    top_k: int = 2
    capacity_factor: float = 1.2
    num_shared_experts: int = 2
    aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    noise_std: float = 0.1

@dataclass
class ModelConfig:
    """Top level configuration for the entire model"""
    architecture: str = "decoder_only" # "decoder_only", "encoder_only", "encoder_decoder"
    vocab_size: int = 32000
    weight_tying: bool = True
    
    # RoPE
    apply_rope: bool = True
    rope_theta: float = 500000.0
    rope_fraction: float = 1.0
    rope_scaling: Optional[str] = None
    rope_in_cross_attn: bool = False

    # Attention
    qk_norm: bool = True
    logit_cap: float = 30.0

    # Inference
    cache_mode: str = "static"
    num_sink_tokens: int = 4
    
    # Training
    use_gradient_checkpointing: bool = True
    use_bf16: bool = True
    
    # Config for each part (if have)
    encoder: Optional[BlockConfig] = None
    decoder: Optional[BlockConfig] = None

    def __post_init__(self):
        if self.encoder:
            self.encoder = BlockConfig(**self.encoder)
        if self.decoder:
            self.decoder = BlockConfig(**self.decoder)


class RotaryPositionalEmbedding(nn.Module):
    """NTK-aware RoPE with proper caching"""
    def __init__(self, d_k: int, max_seq_len: int, theta: float = 500000.0, 
                 rope_fraction: float = 1.0, scaling: Optional[str] = None):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.scaling = scaling
        self.rope_dim = int(self.d_k * rope_fraction)
        self.rope_dim = self.rope_dim - (self.rope_dim % 2)
        
        freqs = 1.0 / (theta ** (torch.arange(0, self.rope_dim, 2).float() / self.rope_dim))
        self.register_buffer("freqs", freqs, persistent=False)
        
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        self._cos_seq_len = 0
    
    def _compute_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        if self.scaling == "ntk" and seq_len > self.max_seq_len:
            alpha = seq_len / self.max_seq_len
            theta = self.theta * alpha ** (self.rope_dim / (self.rope_dim - 2))
            freqs = 1.0 / (theta ** (torch.arange(0, self.rope_dim, 2).float().to(device) / self.rope_dim))
        else:
            freqs = self.freqs.to(device)
        freqs = torch.outer(t, freqs)
        return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0):
        seq_len = k.shape[-2]
        total_len = start_pos + seq_len

        if self.cos_cached is None or total_len > self._cos_seq_len:
            cos, sin = self._compute_cos_sin(total_len * 2, q.device, q.dtype)
            self.register_buffer("cos_cached", cos, persistent=False)
            self.register_buffer("sin_cached", sin, persistent=False)
            self._cos_seq_len = total_len * 2

        # Reshape for broadcasting: (1, 1, seq_len, rope_dim//2)
        cos = self.cos_cached[start_pos : total_len].view(1, 1, seq_len, -1)
        sin = self.sin_cached[start_pos : total_len].view(1, 1, seq_len, -1)

        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)
    
    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # x shape: (batch, heads, seq_len, d_k)
        # cos/sin shape: (1, 1, seq_len, rope_dim//2)

        # Reshape to separate rotation pairs
        x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        x_real, x_imag = x_reshaped[..., 0], x_reshaped[..., 1]

        # Apply rotation (broadcasting now works)
        x_out_real = x_real * cos - x_imag * sin
        x_out_imag = x_real * sin + x_imag * cos

        # Combine back
        x_out = torch.stack([x_out_real, x_out_imag], dim=-1).flatten(start_dim=-2)
        return x_out.type_as(x)


class RMSNorm(nn.Module):
    """Pre-normalization with optional bias"""
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True, 
                 use_bias: bool = False):
        super().__init__()
        self.eps = eps
        self.use_bias = use_bias
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim)) if use_bias else None
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor):
        orig_dtype = x.dtype
        x_fp32 = x.float()
        var = x_fp32.pow(2).mean(-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(var + self.eps)
        
        if self.weight is not None:
            x_normed = self.weight * x_normed
            if self.bias is not None:
                x_normed = x_normed + self.bias
        
        return x_normed.to(orig_dtype)


class SwiGLU(nn.Module):
    """SwiGLU activation for FFN experts"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, 
                 use_bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.w2 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.w3 = nn.Linear(d_ff, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class ExpertChoiceMoE(nn.Module):
    """Expert-choice routing MoE with shared experts"""
    def __init__(self, block_config: BlockConfig, model_config: ModelConfig):
        super().__init__()
        self.block_config = block_config
        self.model_config = model_config
        self.num_experts = block_config.num_experts
        self.top_k = block_config.top_k
        self.capacity_factor = block_config.capacity_factor
        self.aux_loss_coef = block_config.aux_loss_coef
        self.z_loss_coef = block_config.router_z_loss_coef
        self.noise_std = block_config.noise_std
        
        self.gate = nn.Linear(block_config.d_model, block_config.num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLU(block_config.d_model, block_config.d_ff, block_config.dropout, block_config.use_bias)
            for _ in range(block_config.num_experts)
        ])
        
        self.shared_experts = nn.ModuleList([
            SwiGLU(block_config.d_model, block_config.d_ff // 2, block_config.dropout, block_config.use_bias)
            for _ in range(block_config.num_shared_experts)
        ])
    
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, d_model = x.shape
        x_reshaped = x.view(-1, d_model)
        router_logits = self.gate(x_reshaped)
        if self.training:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise
        token_expert_scores, token_expert_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
        token_expert_probs = F.softmax(token_expert_scores, dim=-1)
        final_output = torch.zeros_like(x_reshaped)
        expert_capacity = int((x_reshaped.shape[0] * self.capacity_factor) / self.num_experts)
        for expert_idx, expert in enumerate(self.experts):
            mask = token_expert_indices == expert_idx  # Shape: [num_tokens, top_k]
            token_mask = mask.any(dim=-1)  # Shape: [num_tokens]
            if token_mask.any():
                token_indices = torch.where(token_mask)[0]
                expert_positions = torch.where(mask[token_mask])[1]
                expert_weights = token_expert_probs[token_mask, :].gather(1, expert_positions.unsqueeze(1))

                if token_indices.shape[0] > expert_capacity:
                    dropped_tokens = token_indices[expert_capacity:]
                    token_indices = token_indices[:expert_capacity]
                    expert_weights = expert_weights[:expert_capacity]

                    if dropped_tokens.numel() > 0:
                        final_output.index_add_(0, dropped_tokens, x_reshaped[dropped_tokens])

                tokens = x_reshaped[token_indices]
                expert_out = expert(tokens) * expert_weights
                final_output.index_add_(0, token_indices, expert_out)
        for expert in self.shared_experts:
            final_output = final_output + expert(x_reshaped)
        expert_load = (token_expert_indices == torch.arange(self.num_experts, device=x.device)).float().sum(0)
        if self.num_experts > 1:
            expert_load = (token_expert_indices == torch.arange(self.num_experts, device=x.device)).float().sum(0)
            aux_loss = self.aux_loss_coef * expert_load.std() * self.num_experts
        else:
            aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return final_output.view(batch_size, seq_len, d_model), aux_loss


class StaticCache:
    """Static Cache, High performance using pre-allocation method"""
    def __init__(self, block_config: BlockConfig, batch_size: int, max_seq_len: int, device, dtype):
        self.max_seq_len = max_seq_len
        self.d_k = block_config.d_model // block_config.n_heads

        self.k_cache = torch.zeros(
            (batch_size, block_config.n_layers, block_config.n_kv_heads, max_seq_len, self.d_k),
            dtype=dtype, device=device
        )

        self.v_cache = torch.zeros(
            (batch_size, block_config.n_layers, block_config.n_kv_heads, max_seq_len, self.d_k),
            dtype=dtype, device=device
        )

    def update(self, layer_idx: int, start_pos: int, k_new: torch.Tensor, v_new: torch.Tensor):
        """Update cache at start position `start_pos`"""
        seq_len_new = k_new.shape[-2]
        self.k_cache[:, layer_idx, :, start_pos : start_pos + seq_len_new, :] = k_new
        self.v_cache[:, layer_idx, :, start_pos : start_pos + seq_len_new, :] = v_new

    def get_layer_cache(self, layer_idx: int, seq_len: int):
        """Retrieves the cached K, V for a specific class, up to the current string length only"""
        return (
            self.k_cache[:, layer_idx, :, :seq_len, :],
            self.v_cache[:, layer_idx, :, :seq_len, :]
        )


class AttentionBlock(nn.Module):
    """Multi-head attention with GQA, QK-Norm, sliding window, and Flash Attention"""
    def __init__(self, block_config: BlockConfig, model_config: ModelConfig, is_cross_attention: bool = False, layer_idx: Optional[int] = None):
        super().__init__()
        self.block_config = block_config
        self.model_config = model_config
        self.d_k = block_config.d_model // block_config.n_heads
        self.is_cross_attention = is_cross_attention
        self.layer_idx = layer_idx
        
        self.wq = nn.Linear(block_config.d_model, block_config.d_model, bias=block_config.use_bias)
        self.wk = nn.Linear(block_config.d_model, block_config.n_kv_heads * self.d_k, bias=block_config.use_bias)
        self.wv = nn.Linear(block_config.d_model, block_config.n_kv_heads * self.d_k, bias=block_config.use_bias)
        self.wo = nn.Linear(block_config.d_model, block_config.d_model, bias=block_config.use_bias)
        
        if model_config.qk_norm:
            self.q_norm = RMSNorm(self.d_k, elementwise_affine=True, use_bias=False)
            self.k_norm = RMSNorm(self.d_k, elementwise_affine=True, use_bias=False)
        
        if model_config.apply_rope and not (is_cross_attention and not model_config.rope_in_cross_attn):
            self.rope = RotaryPositionalEmbedding(
                self.d_k, block_config.max_seq_len, model_config.rope_theta,
                model_config.rope_fraction, model_config.rope_scaling
            )
        
        self.sliding_window = block_config.sliding_window if not is_cross_attention else None
        self.dropout = block_config.dropout
    
    def forward(self, xq, xk, xv, mask=None, cache: Optional[StaticCache] = None, start_pos: int = 0):
        batch_size, seq_len_q, _ = xq.shape
        seq_len_k = xk.shape[1]
        
        q = self.wq(xq)
        k = self.wk(xk)
        v = self.wv(xv)
        
        q = q.view(batch_size, seq_len_q, self.block_config.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.block_config.n_kv_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.block_config.n_kv_heads, self.d_k).transpose(1, 2)
        
        if hasattr(self, 'q_norm'):
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        if cache is not None:
            k = k.to(cache.k_cache.dtype)
            v = v.to(cache.v_cache.dtype)
            cache.update(self.layer_idx, start_pos, k, v)
            total_seq_len = start_pos + seq_len_q
            k, v = cache.get_layer_cache(self.layer_idx, total_seq_len)
            seq_len_k = total_seq_len
        
        if self.block_config.n_kv_heads != self.block_config.n_heads:
            num_repeats = self.block_config.n_heads // self.block_config.n_kv_heads
            k = k.unsqueeze(2).repeat_interleave(num_repeats, dim=2)
            v = v.unsqueeze(2).repeat_interleave(num_repeats, dim=2)
            k = k.view(batch_size, self.block_config.n_heads, k.shape[3], self.d_k)
            v = v.view(batch_size, self.block_config.n_heads, v.shape[3], self.d_k)
        
        if hasattr(self, 'rope'):
            q, k = self.rope(q, k)
        
        is_causal = not self.is_cross_attention and mask is None
        
        if mask is None and cache is None and self.sliding_window is not None and k.shape[2] > self.sliding_window:
            actual_k_len = k.shape[2]
            mask = torch.full((seq_len_q, actual_k_len), float('-inf'), device=xq.device)
            mask = torch.triu(mask, diagonal=1)  # Causal
            for i in range(seq_len_q):
                start_idx = max(0, i - self.sliding_window + 1)
                mask[i, :start_idx] = float('-inf')
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal and mask is None
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, -1)
        return self.wo(attn_output)


class DropPath(nn.Module):
    """Stochastic depth regularization"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor
    

class EncoderBlock(nn.Module):
    """A Block of Encoder: Self-Attention -> FFN"""
    def __init__(self, block_config: BlockConfig, model_config: ModelConfig):
        super().__init__()
        # 1. Self-Attention
        self.attention = AttentionBlock(block_config, model_config, is_cross_attention=False)
        self.attention_norm = RMSNorm(block_config.d_model, use_bias=block_config.use_bias)

        # 2. Feed-Forward Network
        self.feed_forward = ExpertChoiceMoE(block_config, model_config)
        self.ffn_norm = RMSNorm(block_config.d_model, use_bias=block_config.use_bias)

        self.drop_path = DropPath(block_config.drop_path) if block_config.drop_path > 0 else nn.Identity()
        self.use_checkpoint = model_config.use_gradient_checkpointing

    def forward(self, x, mask=None):
        def attn_forward(x):
            return self.attention(self.attention_norm(x), self.attention_norm(x), self.attention_norm(x), mask, None)
        
        if self.use_checkpoint and self.training:
            attn_output = torch.utils.checkpoint.checkpoint(attn_forward, x, use_reentrant=False)
        else:
            attn_output = attn_forward(x)
        
        x = x + self.drop_path(attn_output)
        
        def ffn_forward(x):
            return self.feed_forward(self.ffn_norm(x))
        
        if self.use_checkpoint and self.training:
            ffn_output, aux_loss = torch.utils.checkpoint.checkpoint(ffn_forward, x, use_reentrant=False)
        else:
            ffn_output, aux_loss = ffn_forward(x)
        
        return x + self.drop_path(ffn_output), aux_loss
    

class DecoderBlock(nn.Module):
    """
    A Block of Decoder: Masked Self-Attention -> Cross-Attention -> FFN
    """
    def __init__(self, block_config: BlockConfig, model_config: ModelConfig, layer_idx: Optional[int] = None):
        super().__init__()
        # 1. Masked Self-Attention
        self.self_attention = AttentionBlock(block_config, model_config, is_cross_attention=False, layer_idx=layer_idx)
        self.self_attention_norm = RMSNorm(block_config.d_model, use_bias=block_config.use_bias)
        
        # 2. Cross-Attention
        self.cross_attention = AttentionBlock(block_config, model_config, is_cross_attention=True)
        self.cross_attention_norm = RMSNorm(block_config.d_model, use_bias=block_config.use_bias)

        # 3. Feed-Forward Network
        self.feed_forward = ExpertChoiceMoE(block_config, model_config)
        self.ffn_norm = RMSNorm(block_config.d_model, use_bias=block_config.use_bias)
        
        self.drop_path = DropPath(block_config.drop_path) if block_config.drop_path > 0 else nn.Identity()
        self.use_checkpoint = model_config.use_gradient_checkpointing

    def forward(self, x, encoder_output=None, self_attn_mask=None, cross_attn_mask=None, cache: Optional[StaticCache] = None, start_pos: int = 0):
        
        # Self-attention
        def self_attn_forward(x):
            return self.self_attention(self.self_attention_norm(x), self.self_attention_norm(x), 
                                     self.self_attention_norm(x), self_attn_mask, cache, start_pos)
        
        if self.use_checkpoint and self.training:
            attn_output = torch.utils.checkpoint.checkpoint(self_attn_forward, x, use_reentrant=False)
        else:
            attn_output = self_attn_forward(x)
        
        x = x + self.drop_path(attn_output)

        # Cross-attention
        if encoder_output is not None:
            def cross_attn_forward(x, enc_out):
                cross_cache = None
                return self.cross_attention(self.cross_attention_norm(x), enc_out, enc_out, 
                                           cross_attn_mask, cross_cache)
            
            if self.use_checkpoint and self.training:
                cross_output = torch.utils.checkpoint.checkpoint(cross_attn_forward, x, encoder_output, use_reentrant=False)
            else:
                cross_output = cross_attn_forward(x, encoder_output)

            x = x + self.drop_path(cross_output)
        
        # FFN
        def ffn_forward(x):
            return self.feed_forward(self.ffn_norm(x))
        
        if self.use_checkpoint and self.training:
            ffn_output, aux_loss = torch.utils.checkpoint.checkpoint(ffn_forward, x, use_reentrant=False)
        else:
            ffn_output, aux_loss = ffn_forward(x)
        
        x = x + self.drop_path(ffn_output)
        
        return x, aux_loss
    

class TransformerEncoder(nn.Module):
    """Stack Blocks of Encoder. Input token IDs, Output: hidden states."""
    def __init__(self, block_config: BlockConfig, model_config: ModelConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.layers = nn.ModuleList([
            EncoderBlock(block_config, model_config) for _ in range(block_config.n_layers)
        ])
        self.norm = RMSNorm(block_config.d_model, use_bias=block_config.use_bias)
    
    def forward(self, input_ids, attention_mask=None):
        x = self.embed_tokens(input_ids)
        total_aux_loss = 0.0
        
        # Encoder uses bidirectional mask
        mask = attention_mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1).float()
            mask = mask.expand(-1, -1, mask.size(-1), -1)
        
        for layer in self.layers:
            x, aux_loss = layer(x, mask=mask)
            total_aux_loss += aux_loss
        
        return self.norm(x), total_aux_loss / len(self.layers)
    

class TransformerDecoder(nn.Module):
    """Stack Blocks of Decoder."""
    def __init__(self, block_config: BlockConfig, model_config: ModelConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.block_config = block_config
        self.embed_tokens = embed_tokens
        self.layers = nn.ModuleList([
            DecoderBlock(block_config, model_config, layer_idx=i) for i in range(block_config.n_layers)
        ])
        self.norm = RMSNorm(block_config.d_model, use_bias=block_config.use_bias)
        # KV Cache Manager would be managed by TransformerModel 

    def forward(self, input_ids, encoder_output=None, attention_mask=None, cache: Optional[StaticCache] = None, start_pos: int = 0):
        x = self.embed_tokens(input_ids)
        total_aux_loss = 0.0
        
        # Causal mask for self-attention
        seq_len = input_ids.shape[1]
        if cache is not None:
            causal_mask = None
            mask = None
        else:
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)

            # Combine with padding mask if provided
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(1).unsqueeze(1).float()
                mask = mask.expand(-1, -1, seq_len, -1)
                mask = mask.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0) != 0, float('-inf'))
            else:
                mask = causal_mask
        
        for layer in self.layers:            
            x, aux_loss = layer(
                x, 
                encoder_output=encoder_output, 
                self_attn_mask=mask, 
                cross_attn_mask=attention_mask,
                cache=cache,
                start_pos=start_pos
            )
            total_aux_loss += aux_loss
        
        return self.norm(x), total_aux_loss / len(self.layers)
    

class TransformerModel(nn.Module):
    """
    General Transformer model, supporting architectures:
    - Decoder-Only (LLM)
    - Encoder-Only (BERT-like)
    - Encoder-Decoder (T5, BART-like)
    """
    def __init__(self, block_config: BlockConfig, model_config: ModelConfig):
        super().__init__()
        if model_config.architecture not in ["decoder_only", "encoder_only", "encoder_decoder"]:
            raise ValueError(f"Invalid architecture: {model_config.architecture}")
        self.block_config = block_config
        self.model_config = model_config

        # Get the appropriate block_config for embeddings
        embed_dim = block_config.d_model 
        
        # 1. Initialize Token Embeddings (can be shared)
        self.embed_tokens = nn.Embedding(model_config.vocab_size, embed_dim)

        # 2. Build Encoder (if needed)
        self.encoder = None
        if model_config.architecture in ["encoder_only", "encoder_decoder"]:
            self.encoder = TransformerEncoder(model_config.encoder, model_config, self.embed_tokens)

        # 3. Build Decoder (if needed)
        self.decoder = None
        if model_config.architecture in ["decoder_only", "encoder_decoder"]:
            self.decoder = TransformerDecoder(model_config.decoder, model_config, self.embed_tokens)

        # 4. Build Head layers at the output
        self.head = None
        if model_config.architecture == "decoder_only":
            self.head = nn.Linear(block_config.d_model, model_config.vocab_size, bias=False)
            if model_config.weight_tying:
                self.head.weight = self.embed_tokens.weight
                
        elif model_config.architecture == "encoder_decoder":
            self.head = nn.Linear(block_config.d_model, model_config.vocab_size, bias=False)
            if model_config.weight_tying:
                self.head.weight = self.embed_tokens.weight
                
        elif model_config.architecture == "encoder_only":
            # For example, a head layer for a sentence classification task
            self.head = nn.Linear(block_config.d_model, 2)  # Change num_classes as desired
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.block_config.init_std
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, 
                input_ids: torch.Tensor, 
                decoder_input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                decoder_attention_mask: Optional[torch.Tensor] = None,
                cache: Optional[StaticCache] = None,
                start_pos: int = 0):
        
        if self.model_config.architecture == "encoder_only":
            if self.encoder is None: 
                raise ValueError("Encoder is not initialized for this architecture.")
            encoder_output, aux_loss = self.encoder(input_ids, attention_mask)
            logits = self.head(encoder_output)  # For example, take [CLS] token to classify
            return logits, aux_loss

        elif self.model_config.architecture == "decoder_only":
            if self.decoder is None: 
                raise ValueError("Decoder is not initialized for this architecture.")
            decoder_output, aux_loss = self.decoder(
                input_ids, 
                encoder_output=None, 
                attention_mask=attention_mask,
                cache=cache,
                start_pos=start_pos
            )
            logits = self.head(decoder_output)
            return logits, aux_loss
            
        elif self.model_config.architecture == "encoder_decoder":
            if self.encoder is None or self.decoder is None:
                raise ValueError("Encoder/Decoder is not initialized for this architecture.")
            if decoder_input_ids is None:
                raise ValueError("decoder_input_ids is required for encoder-decoder architecture.")

            # Encoder forward pass
            encoder_output, encoder_aux_loss = self.encoder(input_ids, attention_mask)
            
            # Decoder forward pass
            decoder_output, decoder_aux_loss = self.decoder(
                decoder_input_ids, 
                encoder_output, 
                attention_mask=decoder_attention_mask,
                cache=cache,
                start_pos=start_pos
            )
            
            logits = self.head(decoder_output)
            total_aux_loss = encoder_aux_loss + decoder_aux_loss
            return logits, total_aux_loss

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Factory function to create model from YAML file."""
        model_config, block_config = load_config_from_yaml(yaml_path)
        return cls(block_config, model_config)

    def get_num_params(self):
        total = sum(p.numel() for p in self.parameters())
        if self.model_config.weight_tying and hasattr(self, 'head') and hasattr(self.head, 'weight'):
            total -= self.embed_tokens.weight.numel()
        return total
    

def load_config_from_yaml(yaml_path: str) -> Tuple[ModelConfig, BlockConfig]:

    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config_dict = {k: v for k, v in config_dict.items() if v is not None}
    model_config = ModelConfig(**config_dict)
    
    primary_block_config = None
    if model_config.decoder is not None:
        primary_block_config = model_config.decoder
    elif model_config.encoder is not None:
        primary_block_config = model_config.encoder
    else:
        raise ValueError("The loaded configuration has neither an encoder nor a decoder section.")
        
    return model_config, primary_block_config


def export_model_summary_to_file(config_path: str, batch_size: int = 1, seq_len: int = 8192,
                                 output_filename: str = "model_summary.txt"):
    """Export detailed model summary to file"""
    model_config, block_config = load_config_from_yaml(config_path)
    model = TransformerModel.from_yaml(config_path)
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("Modern Transformer Model Summary\n")
        f.write("="*80 + "\n\n")
        
        f.write("Configuration:\n")
        for key, value in block_config.__dict__.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n" + "="*80 + "\n\n")
        for key, value in model_config.__dict__.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n" + "="*80 + "\n\n")
        
        f.write("Architecture:\n")
        f.write("-"*80 + "\n")
        old_stdout, sys.stdout = sys.stdout, f
        print(model)
        sys.stdout = old_stdout
        
        f.write("\n" + "="*80 + "\n\n")
        
        f.write("Trainable Parameters per Layer:\n")
        f.write(f"{'Layer Name':<70} | {'Parameters':>15}\n")
        f.write("-" * 90 + "\n")
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                num_params = param.numel()
                f.write(f"{name:<70} | {num_params:>15,}\n")
        
        f.write("="*90 + "\n")
        total_params = model.get_num_params()
        f.write(f"\nTotal trainable parameters: {total_params:,}\n")
        f.write(f"Total (millions): {total_params / 1e6:.2f}M\n")
            
        bytes_per_param = 2 if model_config.use_bf16 else 4
        weights_memory = total_params * bytes_per_param

        if model_config.architecture == "encoder_decoder":
            total_layers = model_config.decoder.n_layers + model_config.encoder.n_layers
        elif model_config.architecture == "decoder_only":
            total_layers = model_config.decoder.n_layers
        else: # encoder_only
            total_layers = model_config.encoder.n_layers
            
        kv_cache_per_token = 2 * total_layers * block_config.n_kv_heads * (block_config.d_model // block_config.n_heads) * bytes_per_param
        inference_kv_memory = kv_cache_per_token * batch_size * seq_len
        persistent_train_mem = weights_memory * 4
        activation_memory = weights_memory
        total_train_memory = persistent_train_mem + activation_memory
        
        f.write(f"\nMemory Estimates (BF16={model_config.use_bf16}):\n")
        f.write(f"  - Weights Only:              {weights_memory / 1024**3:.2f} GB\n")
        f.write(f"  - Inference (bs={batch_size}, seq={seq_len}):  {(weights_memory + inference_kv_memory) / 1024**3:.2f} GB\n")
        f.write(f"      └─ KV Cache:             {inference_kv_memory / 1024**3:.2f} GB\n")
        f.write(f"  - Training (AdamW):          {total_train_memory / 1024**3:.2f} GB\n")
        f.write(f"      └─ Persistent (w+g+m+v): {persistent_train_mem / 1024**3:.2f} GB\n")
        f.write(f"      └─ Activations (est.):   {activation_memory / 1024**3:.2f} GB\n")
        f.write(f"      Note: Activation memory is a rough lower bound. Real usage can be 2-4x higher\n")
    
    print(f"Summary exported to: {output_filename}")


# --- Testing ---
if __name__ == "__main__":
    import sys # Add this line

    # Check if the user provided a config file path
    if len(sys.argv) < 2:
        print("Error: Please provide the path to a configuration YAML file.")
        print("Usage: python model.py <path_to_your_config>.yaml")
        sys.exit(1) # Exit if no argument is provided

    config_path = sys.argv[1] # Get the file path from the command-line argument

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    batch, seq = 2, 128

    print(f"\n=== Testing with configuration: {config_path} ===")

    try:
        model_config, block_config = load_config_from_yaml(config_path)
        model = TransformerModel(block_config, model_config)

        # Automatically detect the architecture and run the corresponding test
        if model_config.architecture == "decoder_only":
            print("Architecture: Decoder-Only")
            input_ids = torch.randint(0, model_config.vocab_size, (batch, seq))
            logits, aux_loss = model(input_ids)
            print(f"Forward pass successful. Logits: {logits.shape}, Aux loss: {f'{aux_loss:.4f}' if aux_loss is not None else 'N/A'}")

        elif model_config.architecture == "encoder_only":
            print("Architecture: Encoder-Only")
            input_ids = torch.randint(0, model_config.vocab_size, (batch, seq))
            output, aux_loss = model(input_ids)
            print(f"Forward pass successful. Output: {output.shape}, Aux loss: {f'{aux_loss:.4f}' if aux_loss is not None else 'N/A'}")

        elif model_config.architecture == "encoder_decoder":
            print("Architecture: Encoder-Decoder")
            encoder_input_ids = torch.randint(0, model_config.vocab_size, (batch, seq))
            decoder_input_ids = torch.randint(0, model_config.vocab_size, (batch, seq))
            logits, aux_loss = model(
                input_ids=encoder_input_ids,
                decoder_input_ids=decoder_input_ids
            )
            print(f"Forward pass successful. Logits: {logits.shape}, Aux loss: {f'{aux_loss:.4f}' if aux_loss is not None else 'N/A'}")

        else:
            print(f"Architecture '{model_config.architecture}' is not supported in this test script.")

        # Export model summary
        print("\n=== Exporting Model Summary ===")
        output_dir = "architecture"
        os.makedirs(output_dir, exist_ok=True)

        base_filename = f"{config_path.split('/')[-1].replace('.yaml', '')}_summary.txt"
        summary_filepath = os.path.join(output_dir, base_filename)
        export_model_summary_to_file(config_path, batch_size=batch, seq_len=seq,
                                     output_filename=summary_filepath)

    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{config_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")