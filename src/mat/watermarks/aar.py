import scipy.stats
import torch

_HASH_MASK = 0xFFFFFFFF


def _stateless_uniform(offsets: torch.Tensor, seed: int) -> torch.Tensor:
    x = (offsets.to(torch.int64) + seed) & _HASH_MASK
    x = (x ^ (x >> 16)) & _HASH_MASK
    x = (x * 0x85EBCA6B) & _HASH_MASK
    x = (x ^ (x >> 13)) & _HASH_MASK
    x = (x * 0xC2B2AE35) & _HASH_MASK
    x = (x ^ (x >> 16)) & _HASH_MASK
    return x.to(torch.float32) / 4294967296.0


def _hash_pairs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.to(torch.int64) * 2654435761 + b.to(torch.int64)) & _HASH_MASK


class AarWatermark:
    use_argmax_loss = True

    def __init__(self, context_size: int = 2, seed: int = 42, eps: float = 1e-20):
        assert context_size >= 1
        self.k = context_size
        self.seed = int(seed)
        self.eps = eps

    def _uniform(self, context_hashes: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        return _stateless_uniform(_hash_pairs(context_hashes, tokens), self.seed).clamp(
            min=self.eps, max=1 - 1e-6
        )

    def _gumbel(self, context_hashes: torch.Tensor, vocab_size: int, device, dtype) -> torch.Tensor:
        unique_ctx, inverse = torch.unique(context_hashes, return_inverse=True)
        vocab_ids = torch.arange(vocab_size, device=device, dtype=torch.int64)
        u = self._uniform(unique_ctx.view(-1, 1).to(device), vocab_ids.view(1, -1))
        return (-torch.log(-torch.log(u))).to(dtype)[inverse]

    def watermark_logits_argmax(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.LongTensor:
        B, S = input_ids.shape
        V = logits.shape[-1]
        k = self.k
        target = logits.argmax(-1)
        if S < k:
            return target
        ctx = input_ids.unfold(1, k, 1).sum(dim=-1) & _HASH_MASK
        gumbel = self._gumbel(ctx.reshape(-1), V, logits.device, logits.dtype).view(B, S - k + 1, V)
        target[:, k - 1:] = (logits[:, k - 1:] + gumbel).argmax(-1)
        return target

    @torch.no_grad()
    def detect(self, tokens: torch.LongTensor) -> dict:
        z_scores, p_values = [], []
        for seq in tokens.to(torch.int64):
            n = seq.shape[0]
            if n <= self.k:
                z_scores.append(0.0)
                p_values.append(1.0)
                continue
            ctx = seq.unfold(0, self.k, 1).sum(dim=-1) & _HASH_MASK
            u = self._uniform(ctx[:-1], seq[self.k:]).double()
            total = float((-torch.log(1.0 - u)).sum())
            n_scored = u.numel()
            z_scores.append((total - n_scored) / n_scored ** 0.5)
            p_values.append(float(scipy.stats.gamma.sf(total, n_scored, loc=0, scale=1)))
        return {"z_score": torch.tensor(z_scores), "p_value": torch.tensor(p_values)}
