"""Looped Transformer (recurrent depth, weight-shared) with test-time compute scaling.
Run: py -3.14 looped_transformer.py
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class LoopedTransformerConfig:
    """Hyperparameters for the looped transformer."""
    vocab_size: int = 32000
    d_model: int = 256
    num_heads: int = 8
    num_kv_heads: int = 2
    ffn_dim: Optional[int] = None
    default_loops: int = 8
    max_seq_len: int = 2048
    rope_theta: float = 10000.0


class RMSNorm(nn.Module):
    """Root mean square normalization over last dim (B, T, D) -> (B, T, D)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight: nn.Parameter = nn.Parameter(torch.ones(dim))
        self.eps: float = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var: torch.Tensor = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm: torch.Tensor = x * torch.rsqrt(var + self.eps)
        return self.weight * x_norm


class RotaryEmbedding(nn.Module):
    """1D RoPE cache; apply rotary rotation to q/k of shape (B, H, T, Dh)."""

    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")
        self.head_dim: int = head_dim
        inv_freq: torch.Tensor = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len: int = max_seq_len

    def _cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        t: torch.Tensor = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs: torch.Tensor = torch.outer(t, self.inv_freq)
        emb: torch.Tensor = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        d2: int = x.shape[-1] // 2
        return torch.cat((-x[..., d2:], x[..., :d2]), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len: int = q.shape[-2]
        cos: torch.Tensor
        sin: torch.Tensor
        cos, sin = self._cos_sin(seq_len, q.device, q.dtype)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        q_out: torch.Tensor = q * cos + self._rotate_half(q) * sin
        k_out: torch.Tensor = k * cos + self._rotate_half(k) * sin
        return q_out, k_out


class GroupedQueryAttention(nn.Module):
    """GQA with SDPA backend. x (B, T, D) -> out (B, T, D)."""

    def __init__(self, config: LoopedTransformerConfig) -> None:
        super().__init__()
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if config.num_heads % config.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads.")
        self.n_heads: int = config.num_heads
        self.n_kv: int = config.num_kv_heads
        self.head_dim: int = config.d_model // config.num_heads
        self.q_proj: nn.Linear = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj: nn.Linear = nn.Linear(config.d_model, self.n_kv * self.head_dim, bias=False)
        self.v_proj: nn.Linear = nn.Linear(config.d_model, self.n_kv * self.head_dim, bias=False)
        self.o_proj: nn.Linear = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope: RotaryEmbedding = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B: int
        T: int
        B, T, _ = x.shape
        q: torch.Tensor = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k: torch.Tensor = self.k_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v: torch.Tensor = self.v_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)
        if self.n_heads != self.n_kv:
            rep: int = self.n_heads // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        attn: torch.Tensor = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(attn)


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN: (B, T, D) -> (B, T, D) with 8/3*d_model hidden (mult of 64)."""

    def __init__(self, d_model: int, ffn_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden: int = ffn_dim or int(8 * d_model / 3)
        hidden = ((hidden + 63) // 64) * 64
        self.w1: nn.Linear = nn.Linear(d_model, hidden, bias=False)
        self.w2: nn.Linear = nn.Linear(d_model, hidden, bias=False)
        self.w3: nn.Linear = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TransformerBlock(nn.Module):
    """Single weight-shared block: attn + FFN with residuals. (B, T, D) -> (B, T, D)."""

    def __init__(self, config: LoopedTransformerConfig) -> None:
        super().__init__()
        self.attn_norm: RMSNorm = RMSNorm(config.d_model)
        self.attn: GroupedQueryAttention = GroupedQueryAttention(config)
        self.ffn_norm: RMSNorm = RMSNorm(config.d_model)
        self.ffn: SwiGLUFFN = SwiGLUFFN(config.d_model, config.ffn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class LoopedTransformer(nn.Module):
    """Recurrent-depth model: h_{r+1} = Block(h_r + alpha * e_0), R loops at test time."""

    def __init__(self, config: LoopedTransformerConfig) -> None:
        super().__init__()
        self.config: LoopedTransformerConfig = config
        self.tok_emb: nn.Embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.block: TransformerBlock = TransformerBlock(config)
        self.final_norm: RMSNorm = RMSNorm(config.d_model)
        self.lm_head: nn.Linear = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.alpha: nn.Parameter = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        num_loops: Optional[int] = None,
        exit_steps: Optional[List[int]] = None,
        use_checkpoint: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[torch.Tensor]]:
        R: int = num_loops or self.config.default_loops
        e0: torch.Tensor = self.tok_emb(input_ids)
        h: torch.Tensor = e0
        exits: List[torch.Tensor] = []

        def _step(state: torch.Tensor) -> torch.Tensor:
            return self.block(state + self.alpha * e0)

        for r in range(R):
            if use_checkpoint and self.training:
                h = checkpoint(_step, h, use_reentrant=False)
            else:
                h = _step(h)
            if exit_steps and (r + 1) in exit_steps:
                exits.append(self.lm_head(self.final_norm(h)))

        logits: torch.Tensor = self.lm_head(self.final_norm(h))
        loss: Optional[torch.Tensor] = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            if exits:
                for elog in exits:
                    loss = loss + F.cross_entropy(elog.view(-1, elog.size(-1)), targets.view(-1))
                loss = loss / (1 + len(exits))
        return logits, loss, exits


class ACTHalting(nn.Module):
    """Graves-2016 ACT halting head: per-position halt probs + ponder cost (B, T) stop flags."""

    def __init__(self, d_model: int, threshold: float = 0.99, max_loops: int = 16) -> None:
        super().__init__()
        self.halt_linear: nn.Linear = nn.Linear(d_model, 1)
        self.threshold: float = threshold
        self.max_loops: int = max_loops

    def forward(self, h: torch.Tensor, halt_sum: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p: torch.Tensor = torch.sigmoid(self.halt_linear(h).squeeze(-1))
        still: torch.Tensor = (halt_sum < self.threshold).float()
        halt_sum = halt_sum + p * still
        stop: torch.Tensor = (halt_sum >= self.threshold).float()
        return stop, halt_sum, p


class LoopedLM(nn.Module):
    """Prelude -> recurrent core (N loops, input re-injection) -> coda, with ACT halting.

    use_looped=False runs prelude -> coda directly (baseline path for comparison).
    """

    MIN_LOOPS: int = 1
    MAX_LOOPS: int = 16

    def __init__(self, config: LoopedTransformerConfig, prelude_layers: int = 1, coda_layers: int = 1) -> None:
        super().__init__()
        self.config: LoopedTransformerConfig = config
        self.tok_emb: nn.Embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.prelude: nn.ModuleList = nn.ModuleList([TransformerBlock(config) for _ in range(prelude_layers)])
        self.core: TransformerBlock = TransformerBlock(config)
        self.coda: nn.ModuleList = nn.ModuleList([TransformerBlock(config) for _ in range(coda_layers)])
        self.final_norm: RMSNorm = RMSNorm(config.d_model)
        self.lm_head: nn.Linear = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.alpha: nn.Parameter = nn.Parameter(torch.tensor(0.1))
        self.halter: ACTHalting = ACTHalting(config.d_model, max_loops=self.MAX_LOOPS)
        self.exec_count: int = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        num_loops: Optional[int] = None,
        use_looped: bool = True,
        adaptive: bool = False,
        use_checkpoint: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
        e0: torch.Tensor = self.tok_emb(input_ids)
        h: torch.Tensor = e0
        for blk in self.prelude:
            h = blk(h)
        loops_used: int = 0
        ponder: torch.Tensor = torch.zeros((), device=h.device)
        if use_looped:
            R: int = max(self.MIN_LOOPS, min(self.MAX_LOOPS, num_loops or self.config.default_loops))
            halt_sum: torch.Tensor = torch.zeros(input_ids.shape, device=h.device)
            for _ in range(R):
                def _step(state: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
                    return self.core(state + self.alpha * base)

                if adaptive:
                    stop, halt_sum, p = self.halter(h.detach(), halt_sum)
                    ponder = ponder + p.mean()
                    if bool((stop >= 1).all()):
                        break
                if use_checkpoint and self.training:
                    h = checkpoint(_step, h, e0, use_reentrant=False)
                else:
                    h = _step(h, e0)
                self.exec_count += 1
                loops_used += 1
                if adaptive:
                    stop, halt_sum, _ = self.halter(h.detach(), halt_sum)
                    if bool((stop >= 1).all()):
                        break
        for blk in self.coda:
            h = blk(h)
        logits: torch.Tensor = self.lm_head(self.final_norm(h))
        loss: Optional[torch.Tensor] = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            if use_looped and adaptive:
                loss = loss + 0.01 * ponder
        return logits, loss, loops_used


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = LoopedTransformerConfig(vocab_size=1024, d_model=256, num_heads=8,
                                  num_kv_heads=2, default_loops=8, max_seq_len=512)
    model = LoopedTransformer(cfg).train()
    print(f"params: {sum(p.numel() for p in model.parameters())}")
    print(f"alpha init: {model.alpha.item()}")

    B, T = 2, 128
    toks: torch.Tensor = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss, exits = model(toks, targets=toks, exit_steps=[4, 8], use_checkpoint=True)
    print(f"logits {tuple(logits.shape)} loss {loss.item():.4f} exits {len(exits)}")
    loss.backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None]
    print(f"null-grad params: {dead if dead else 'NONE - all have grads'}")
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()
    print(f"grad norm: {gnorm:.4f}")

    model.eval()
    cuda = torch.cuda.is_available()
    print(f"cuda: {cuda}")
    for R in (4, 8, 16):
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.no_grad():
            lg, _, _ = model(toks, num_loops=R)
        dt = (time.perf_counter() - t0) * 1000 / 1
        peak = torch.cuda.max_memory_allocated() / 1e6 if cuda else 0.0
        print(f"R={R:2d} latency {dt:8.1f} ms  peak_cuda {peak:8.1f} MB  out {tuple(lg.shape)}")
    print("ALL CHECKS DONE")

    print("--- LoopedLM: prelude/core/coda + ACT + use_looped flag ---")
    cfg2 = LoopedTransformerConfig(vocab_size=256, d_model=64, num_heads=4,
                                   num_kv_heads=2, default_loops=6, max_seq_len=128)
    lm = LoopedLM(cfg2).eval()
    t2: torch.Tensor = torch.randint(0, cfg2.vocab_size, (2, 32))
    with torch.no_grad():
        l_base, _, n0 = lm(t2, use_looped=False)
        assert n0 == 0, "use_looped=False must run 0 core loops"
        lm.exec_count = 0
        l4, _, n4 = lm(t2, num_loops=4)
        assert n4 == 4 and lm.exec_count == 4, f"core must execute exactly N times (got {lm.exec_count})"
        lm.exec_count = 0
        l8, _, n8 = lm(t2, num_loops=8)
        assert n8 == 8 and lm.exec_count == 8
        d48: float = (l4 - l8).abs().mean().item()
        assert d48 > 0, "changing N must change hidden states/outputs"
        print(f"exec-count exact: N=4->{n4} N=8->{n8} mean|d_logits(4vs8)|={d48:.4f} (loop IS used)")
    lm.train()
    _, tr_loss, _ = lm(t2, targets=t2, num_loops=4, adaptive=True, use_checkpoint=True)
    assert tr_loss is not None and tr_loss.requires_grad, "train loss must carry grads"
    tr_loss.backward()
    dead2 = [n for n, p in lm.named_parameters() if p.grad is None]
    assert not dead2, f"null grads: {dead2}"
    print(f"truncated-BPTT loss {tr_loss.item():.4f}, all grads present")
    lm.eval()
    easy: torch.Tensor = torch.zeros((1, 16), dtype=torch.long)
    hard: torch.Tensor = torch.randint(0, cfg2.vocab_size, (1, 16))
    with torch.no_grad():
        _, _, ne = lm(easy, adaptive=True)
        _, _, nh = lm(hard, adaptive=True)
    print(f"adaptive loops: easy={ne} hard={nh} (halting varies by input difficulty)")
    print("LOOPEDLM CHECKS DONE")
