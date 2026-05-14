from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def run_clustering(X, n_clusters=3):

    kmeans = KMeans(n_clusters=n_clusters)

    clusters = kmeans.fit_predict(X)

    pca = PCA(n_components=2)

    components = pca.fit_transform(X)

    return {
        'x': components[:, 0].tolist(),
        'y': components[:, 1].tolist(),
        'clusters': clusters.tolist()
    }