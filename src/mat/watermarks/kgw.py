from typing import Optional

import torch

from ..rng import stateless_uniform

SEEDING_SCHEMES = {
    "simple_1": dict(context_width=1, hash_key=15485863),
}


class KGWWatermark:
    def __init__(
        self,
        tokenizer,
        gamma: float = 0.25,
        delta: float = 2.0,
        seeding_scheme: str = "simple_1",
        top_k: Optional[int] = None,
    ):
        scheme = SEEDING_SCHEMES[seeding_scheme]
        self.k = scheme["context_width"]
        self.hash_key = scheme["hash_key"]
        self.vocab_size = len(tokenizer.get_vocab())
        self.gamma = gamma
        self.delta = delta
        self.top_k = top_k
        if top_k is not None and top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        self.special_token_ids = [
            t
            for t in (tokenizer.eos_token_id, tokenizer.bos_token_id,
                      tokenizer.pad_token_id, tokenizer.unk_token_id)
            if t is not None
        ]

    def _context_hashes(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        return input_ids.unfold(-1, self.k, 1).sum(dim=-1)

    def _greenlist_mask(self, hashes: torch.LongTensor) -> torch.BoolTensor:
        token_ids = torch.arange(self.vocab_size, device=hashes.device).view(1, 1, -1)
        combined = (hashes.unsqueeze(-1).int() * 2654435761 + token_ids.int()).to(torch.int32)
        mask = stateless_uniform(combined, self.hash_key) < self.gamma
        for tid in self.special_token_ids:
            if tid < self.vocab_size:
                mask[:, :, tid] = False
        return mask

    def _greenlist_pointwise(self, hashes: torch.LongTensor, tokens: torch.LongTensor) -> torch.BoolTensor:
        combined = (hashes.int() * 2654435761 + tokens.int()).to(torch.int32)
        mask = stateless_uniform(combined, self.hash_key) < self.gamma
        for tid in self.special_token_ids:
            mask = mask & (tokens != tid)
        return mask

    def watermark_logits(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.FloatTensor:
        mask = self._greenlist_mask(self._context_hashes(input_ids))
        target = logits[..., self.k - 1:, : mask.shape[-1]]
        if self.top_k is not None:
            _, topk_idx = target.topk(min(self.top_k, target.shape[-1]), dim=-1)
            in_topk = torch.zeros_like(mask)
            in_topk.scatter_(-1, topk_idx, True)
            mask = mask & in_topk
        target[mask] += self.delta
        return logits

    @torch.no_grad()
    def detect(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None) -> dict:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids).bool()
        hashes = self._context_hashes(input_ids)
        tokens = input_ids[:, self.k:]
        if input_ids.shape[0] * input_ids.shape[1] <= 500_000:
            mask = self._greenlist_mask(hashes)
            B, T = tokens.shape
            batch_idx = torch.arange(B, device=input_ids.device).unsqueeze(1).expand(-1, T)
            time_idx = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
            is_green = mask[batch_idx, time_idx, tokens]
        else:
            is_green = self._greenlist_pointwise(hashes[:, :-1], tokens)

        scored = _mask_repeated_kgrams(input_ids, self.k)[:, self.k:] * attention_mask[:, self.k:]
        n_scored = scored.sum(dim=1)
        n_green = (is_green * scored).sum(dim=1).float()
        z = (n_green - self.gamma * n_scored) / torch.sqrt(self.gamma * (1 - self.gamma) * n_scored)
        p = 0.5 * torch.erfc(z / torch.sqrt(torch.tensor(2.0, device=z.device)))
        return {"z_score": z, "p_value": p}


def _mask_repeated_kgrams(input_ids: torch.LongTensor, k: int) -> torch.BoolTensor:
    B, L = input_ids.shape
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    for b in range(B):
        seen = set()
        seq = input_ids[b].tolist()
        for i in range(L - k):
            gram = tuple(seq[i: i + k + 1])
            if gram in seen:
                mask[b, i + k] = False
            else:
                seen.add(gram)
    return mask
