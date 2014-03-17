import numpy as np
from numpy import pi

def non_maximal_suppresion(mag, orient):
    """Non Maximal suppression of gradient magnitude and orientation."""
    # bin orientations into 4 discrete directions
    abin = ((orient + pi) * 4 / pi + 0.5).astype('int') % 4

    mask = np.zeros(mag.shape, dtype='bool')
    mask[1:-1, 1:-1] = True
    edge_map = np.zeros(mag.shape, dtype='bool')
    offsets = ((1, 0), (1, 1), (0, 1), (-1, 1))
    for a, (di, dj) in zip(range(4), offsets):
        cand_idx = np.nonzero(np.logical_and(abin == a, mask))
        for i, j in zip(*cand_idx):
            if mag[i, j] > mag[i + di, j + dj] and \
               mag[i, j] > mag[i - di, j - dj]:
                edge_map[i, j] = True
    return edge_map
