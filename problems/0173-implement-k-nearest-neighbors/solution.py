import numpy as np

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    n = len(points)
    points = np.array(points)
    distances = np.zeros(n)

    for i, p in enumerate(points):
        distances[i] = euclidean(p, query_point)
    indices = np.argsort(distances, kind="stable")
    sorted_neighbors = points[indices]
    sorted_neighbors = list(map(tuple, sorted_neighbors))
    top_k = sorted_neighbors[:k]
    
    return top_k
        

def euclidean(p1, p2):
    return np.sum((p1 - p2)**2)