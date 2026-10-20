cimport cython
from libc.math cimport logf
from libc.stdlib cimport free, malloc


@cython.boundscheck(False)
@cython.wraparound(False)
def levenshtein(const long[:] tokens, const float[:, :] key, float gamma=0.0):
    cdef Py_ssize_t n = tokens.shape[0]
    cdef Py_ssize_t m = key.shape[0]
    cdef Py_ssize_t i, j
    cdef float best, cand
    cdef float *prev = <float *> malloc((m + 1) * sizeof(float))
    cdef float *curr = <float *> malloc((m + 1) * sizeof(float))
    cdef float *swap
    if prev == NULL or curr == NULL:
        free(prev)
        free(curr)
        raise MemoryError()
    try:
        for j in range(m + 1):
            prev[j] = j * gamma
        for i in range(1, n + 1):
            curr[0] = i * gamma
            for j in range(1, m + 1):
                best = prev[j] + gamma
                cand = curr[j - 1] + gamma
                if cand < best:
                    best = cand
                cand = prev[j - 1] + logf(1.0 - key[j - 1, tokens[i - 1]])
                if cand < best:
                    best = cand
                curr[j] = best
            swap = prev
            prev = curr
            curr = swap
        return prev[m]
    finally:
        free(prev)
        free(curr)
