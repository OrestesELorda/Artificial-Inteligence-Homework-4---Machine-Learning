# kmeans_custom.py
import numpy as np

class KMeansCustom:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = int(n_clusters)
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None

    def _init_centroids(self, X):
        rng = np.random.default_rng(self.random_state)
        # simple random initialization (could be replaced with kmeans++)
        idx = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
        return X[idx].astype(float)

    def _assign_clusters(self, X, centroids):
        # compute distances to centroids and assign
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)  # shape (n_samples, k)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            points = X[labels == k]
            if len(points) == 0:
                # empty cluster: reinitialize to random point
                new_centroids[k] = X[np.random.randint(0, X.shape[0])]
            else:
                new_centroids[k] = points.mean(axis=0)
        return new_centroids

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        centroids = self._init_centroids(X)
        for it in range(self.max_iter):
            labels = self._assign_clusters(X, centroids)
            new_centroids = self._update_centroids(X, labels)
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift <= self.tol:
                break
        self.centroids = centroids
        self.labels_ = labels
        return self

    def predict(self, X):
        return self._assign_clusters(np.asarray(X, dtype=float), self.centroids)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


def wcss(X, labels, centroids):
    total = 0.0
    for k in range(centroids.shape[0]):
        points = X[labels == k]
        if len(points) == 0:
            continue
        total += np.sum((points - centroids[k]) ** 2)
    return total


def compute_wcss_for_ks(X, ks=range(2, 9), random_state=42):
    wcss_list = []
    for k in ks:
        km = KMeansCustom(n_clusters=k, max_iter=200, tol=1e-4, random_state=random_state)
        labels = km.fit_predict(X)
        wcss_val = wcss(X, labels, km.centroids)
        wcss_list.append(wcss_val)
    return wcss_list


def choose_k_elbow(wcss_list):
    """
    Simple elbow detection using the second derivative heuristic:
    choose k where the drop in WCSS decreases the most (max second difference index).
    ks correspond to 2..(len(wcss_list)+1)
    """
    # compute discrete second derivative
    w = np.array(wcss_list)
    if len(w) < 3:
        return 2
    first_diff = np.diff(w)
    second_diff = np.diff(first_diff)
    # choose index of minimum second_diff (largest curvature) -> add 2 because ks start at 2
    idx = int(np.argmin(second_diff)) + 2
    return idx
