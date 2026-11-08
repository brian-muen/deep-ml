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
    query_point = np.asarray(query_point)
    points = np.asarray(points)

    distances = np.sum((query_point - points)**2, axis=1)
    indices = np.argsort(distances, kind="stable")
    sorted_neighbors = points[indices]

    return list(map(tuple, sorted_neighbors[:k]))