# Cell 1
import numpy as np

def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
    points = np.asarray(points, dtype=float)              
    centroids = np.asarray(initial_centroids, dtype=float) 
    
    # centroids shape: (k, d)
    # points shape: (n, d)
    
    for _ in range(max_iterations):
        clusters = find_clusters(points, centroids)
        new_centroids = calculate_centroids(points, clusters, k, centroids)
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
        
    return centroids
    
    
def find_clusters(points, centroids):
    diffs = points[:, None, :] - centroids[None, :, :] # (n, k, d)
    sq_dists = np.sqrt((diffs ** 2).sum(axis=2))   
    
    return sq_dists.argmin(axis=1)
        
def calculate_centroids(points, clusters, k, centroids):
    new_centroids = centroids.copy()
    for idx, centroid in enumerate(centroids):
        cluster_points = points[clusters == idx]         
        
        if len(cluster_points) > 0:
            new_centroids[idx] = cluster_points.mean(axis=0)
        
    return new_centroids