import numpy as np
from utils import colors, get_color_prob


def imagen_a_puntos_rgb(im):
    return im.reshape(-1, 3).astype(np.float64)


class Kmeans:
    def __init__(self, X, K=3):
        self.X = np.asarray(X, dtype=np.float64)
        self.K = int(K)
        self.n = len(self.X)
        self.centroids = None
        self.labels = None

    def fit(self, max_iter=100, semilla=42):
        np.random.seed(semilla)
        self.centroids = self.X[np.random.choice(self.n, self.K, replace=False)].copy()
        for _ in range(max_iter):
            antes = self.centroids.copy()
            self.labels = ((self.X[:, None] - self.centroids) ** 2).sum(2).argmin(1)
            for k in range(self.K):
                m = self.labels == k
                self.centroids[k] = self.X[m].mean(0) if m.any() else self.X[np.random.randint(self.n)]
            if np.linalg.norm(self.centroids - antes) < 1e-4:
                break
        return self

    def withinClassDistance(self):
        return float(((self.X - self.centroids[self.labels]) ** 2).sum())

    def find_bestK(self, max_K=8):
        if max_K < 3:
            return 2
        wcd = [Kmeans(self.X, K).fit().withinClassDistance() for K in range(2, max_K + 1)]
        for i, (a, b) in enumerate(zip(wcd, wcd[1:])):
            if a > 0 and (a - b) / a < 0.2:
                return i + 2
        return max_K

    def get_color(self):
        assert self.centroids is not None, "Crida fit() primer."
        out = []
        for c in self.centroids:
            rgb = np.clip(c, 0, 255).astype(np.uint8).reshape(1, 1, 3)
            out.append(str(colors[np.argmax(get_color_prob(rgb))]))
        return out
