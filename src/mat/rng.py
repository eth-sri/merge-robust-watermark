import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:

    @triton.jit
    def _uniform_kernel(offsets_ptr, out_ptr, n_elements, seed, BLOCK: tl.constexpr):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = idx < n_elements
        offs = tl.load(offsets_ptr + idx, mask=mask).to(tl.int32)
        tl.store(out_ptr + idx, tl.rand(seed, offs), mask=mask)


def stateless_uniform(offsets: torch.Tensor, seed: int, block_size: int = 1024) -> torch.Tensor:
    if triton is None or not offsets.is_cuda:
        raise RuntimeError(
            "The KGW green list is defined by Triton's Philox RNG on CUDA. "
            "Run training and detection on a CUDA device with `triton` installed."
        )
    flat = offsets.contiguous().view(-1)
    if flat.dtype not in (torch.int32, torch.int64):
        flat = flat.to(torch.int64)
    out = torch.empty(flat.numel(), device=flat.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(flat.numel(), meta["BLOCK"]),)
    _uniform_kernel[grid](flat, out, flat.numel(), seed, BLOCK=block_size)
    return out.view(offsets.shape)
