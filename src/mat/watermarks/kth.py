import numpy as np
import torch

_levenshtein_fast = None
try:
    from levenshtein_rust import detect as _rust_detect, permutation_test_parallel as _rust_permutation_test
except ImportError:
    _rust_detect = _rust_permutation_test = None
    try:
        import pyximport
        pyximport.install(language_level=3, setup_args={"include_dirs": np.get_include()})
        from .levenshtein import levenshtein as _levenshtein_fast
    except ImportError:
        pass


def _levenshtein(tokens: np.ndarray, xi: np.ndarray, gamma: float = 0.0) -> float:
    if _levenshtein_fast is not None:
        return _levenshtein_fast(tokens, xi, gamma)
    n, m = len(tokens), len(xi)
    A = np.zeros((n + 1, m + 1), dtype=np.float64)
    A[:, 0] = np.arange(n + 1) * gamma
    A[0, :] = np.arange(m + 1) * gamma
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            A[i, j] = min(A[i - 1, j] + gamma, A[i, j - 1] + gamma,
                          A[i - 1, j - 1] + np.log(1.0 - xi[j - 1, tokens[i - 1]]))
    return float(A[n, m])


class KTHWatermark:
    def __init__(self, vocab_size: int, key_size: int = 256, seed: int = 42, n_runs: int = 100, eps: float = 1e-20):
        self.key_size = key_size
        self.n_runs = n_runs
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        uniform = torch.rand((key_size, vocab_size), generator=g, dtype=torch.float32).clamp(min=eps, max=1 - 1e-6)
        self.gumbel = -torch.log(-torch.log(uniform))
        self.xi = uniform.numpy().astype(np.float32)
        self.null_rng = np.random.default_rng(int(seed) + 1)

    def _min_cost(self, tokens: np.ndarray, xi: np.ndarray) -> float:
        n, k = self.key_size, len(tokens)
        positions = np.arange(k)
        return min(_levenshtein(tokens, xi[(j + positions) % n]) for j in range(n))

    @torch.no_grad()
    def detect(self, tokens: torch.LongTensor) -> dict:
        z_scores, p_values = [], []
        n = self.key_size
        for seq in tokens:
            tok = seq.cpu().numpy().astype(np.int64)
            unique, compact = np.unique(tok, return_inverse=True)
            compact = compact.astype(np.int64)
            key = np.ascontiguousarray(self.xi[:, unique])
            if _rust_detect is not None:
                cost = float(_rust_detect(compact, n, len(tok), key, 0.0))
                p_value = float(_rust_permutation_test(compact, n, len(tok), len(unique), cost, self.n_runs))
            else:
                cost = self._min_cost(compact, key)
                hits = sum(
                    self._min_cost(compact, self.null_rng.random((n, len(unique)), dtype=np.float32)) <= cost
                    for _ in range(self.n_runs)
                )
                p_value = (hits + 1.0) / (self.n_runs + 1.0)
            z_scores.append(-cost)
            p_values.append(p_value)
        return {"z_score": torch.tensor(z_scores), "p_value": torch.tensor(p_values)}
