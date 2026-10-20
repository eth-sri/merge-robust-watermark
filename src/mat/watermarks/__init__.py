from .aar import AarWatermark
from .gaussmark import GaussMarkSecret
from .kgw import KGWWatermark
from .kth import KTHWatermark


def build_watermark(cfg, tokenizer):
    if cfg.type == "kgw":
        return KGWWatermark(tokenizer, gamma=cfg.gamma, delta=cfg.delta,
                            seeding_scheme=cfg.seeding_scheme, top_k=cfg.top_k)
    if cfg.type == "aar":
        return AarWatermark(context_size=cfg.aar_context_size, seed=cfg.aar_seed)
    if cfg.type == "kth":
        return KTHWatermark(vocab_size=len(tokenizer), key_size=cfg.kth_key_size,
                            seed=cfg.kth_seed, n_runs=cfg.kth_n_runs)
    raise ValueError(f"Unknown watermark type: {cfg.type}")
