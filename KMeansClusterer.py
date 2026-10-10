import numpy as np


class KMeansClusterer:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.n_iter_ = 0

    def fit(self, data):
        data = np.asarray(data, dtype=float)
        n = len(data)
        if n == 0:
            self.centroids = np.empty((0, 2))
            self.labels = np.array([])
            return self.labels

        k = min(self.n_clusters, n)

        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(n, size=k, replace=False)
        self.centroids = data[idx].copy()

        self.n_iter_ = 0
        for it in range(self.max_iter):
            self.n_iter_ = it + 1
            distances = np.linalg.norm(data[:, None, :] - self.centroids[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.zeros_like(self.centroids)
            for i in range(k):
                members = data[labels == i]
                if len(members) > 0:
                    new_centroids[i] = members.mean(axis=0)
                else:
                    new_centroids[i] = data[rng.integers(0, n)]

            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            self.labels = labels

            if shift < self.tol:
                break

        return self.labels

    def predict(self, point):
        point = np.asarray(point, dtype=float).ravel()
        distances = np.linalg.norm(self.centroids - point, axis=1)
        return int(np.argmin(distances))