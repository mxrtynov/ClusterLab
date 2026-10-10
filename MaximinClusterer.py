import numpy as np


class MaximinClusterer:

    def __init__(self, threshold=1.2, max_clusters=3):
        self.threshold = threshold
        self.max_clusters = max_clusters
        self.centroids = None
        self.labels = None

    def fit(self, data):
        data = np.asarray(data, dtype=float)
        n = len(data)

        if n == 0:
            self.centroids = np.empty((0, 2))
            self.labels = np.array([])
            return self.labels

        if n == 1:
            self.centroids = data.copy()
            self.labels = np.array([0])
            return self.labels

        dist = np.linalg.norm(data[:, None, :] - data[None, :, :], axis=2)

        first = int(np.argmax(dist.sum(axis=1)))
        centroids = [data[first]]
        min_dist = dist[first].copy()
        min_dist[first] = -1.0

        while True:
            if len(centroids) >= self.max_clusters:
                break
            max_min = float(np.max(min_dist))
            if max_min < self.threshold:
                break
            idx = int(np.argmax(min_dist))
            centroids.append(data[idx])
            min_dist = np.minimum(min_dist, dist[idx])
            min_dist[idx] = -1.0

        self.centroids = np.array(centroids)

        labels = []
        for p in data:
            d = np.linalg.norm(self.centroids - p, axis=1)
            labels.append(int(np.argmin(d)))
        self.labels = np.array(labels)

        return self.labels

    def predict(self, point):
        point = np.asarray(point, dtype=float).ravel()
        d = np.linalg.norm(self.centroids - point, axis=1)
        return int(np.argmin(d))