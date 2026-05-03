from collections import Counter

import numpy as np
from utils import rgb2gray
from utils_data import read_dataset as _rd


def read_dataset(gt_json="./images/gt.json", **kw):
    return _rd(gt_json=gt_json, **kw)


def _vec(im):
    return (rgb2gray(im) if im.ndim == 3 else im).ravel().astype(np.float64)


class KNN:
    def __init__(self, X, y, k=5):
        self.k = k
        self.y = np.asarray(y)
        self.X = np.array([_vec(im) for im in X])

    def predict(self, T):
        out = []
        for t in T:
            d = np.linalg.norm(self.X - _vec(t), axis=1)
            out.append(Counter(self.y[np.argsort(d)[: self.k]]).most_common(1)[0][0])
        return np.array(out, dtype=self.y.dtype)
