import numpy as np
from tqdm import tqdm
from numba import jit


@jit(nopython=True)
def mrf_kernel(mrf_old, mrf_gamma, mrf, neighborhood_size):
    for r in range(mrf.shape[0]):
        for c in range(mrf.shape[1]):
            pixel_propabilities = mrf_old[
                r,
                c,
            ]
            neighbor_cnt = 0

            m = np.zeros(15)

            n_row_start = max(0, r - neighborhood_size)
            n_row_end = min(mrf.shape[0], r + neighborhood_size + 1)

            n_col_start = max(0, c - neighborhood_size)
            n_col_end = min(mrf.shape[1], c + neighborhood_size + 1)

            for n_row in range(n_row_start, n_row_end):
                for n_col in range(n_col_start, n_col_end):
                    if n_row != r or n_col != c:  # skip self
                        m[mrf_old[n_row, n_col, :].argmax()] += 1
                        neighbor_cnt += 1

            gibs = np.exp(-mrf_gamma * (neighbor_cnt - m))
            mrf_probabilities = gibs * pixel_propabilities
            mrf_probabilities /= np.sum(mrf_probabilities)
            mrf[
                r,
                c,
            ] = mrf_probabilities

    return mrf


def MRF(original, mrf_iterations=5, mrf_gamma=0.3, neighborhood_size=11):
    mrf_old = np.array(original)
    mrf = np.zeros(np.shape(original))

    for i in tqdm(range(mrf_iterations)):
        mrf = mrf_kernel(mrf_old, mrf_gamma, mrf, neighborhood_size)
        mrf_old = mrf

    return mrf
