def get_bounding_box_center(bounding_box):
    min_point, max_point = bounding_box

    center_x = (min_point[0] + max_point[0]) / 2
    center_y = (min_point[1] + max_point[1]) / 2
    center_z = (min_point[2] + max_point[2]) / 2

    return (center_x, center_y, center_z)


def get_xlen(bounding_box):
    min_point, max_point = bounding_box
    return max_point[0] - min_point[0]


def get_ylen(bounding_box):
    min_point, max_point = bounding_box
    return max_point[1] - min_point[1]


def get_zlen(bounding_box):
    min_point, max_point = bounding_box
    return max_point[2] - min_point[2]


def get_xmin(bounding_box):
    min_point, _ = bounding_box
    return min_point[0]


def get_ymin(bounding_box):
    min_point, _ = bounding_box
    return min_point[1]


def get_zmin(bounding_box):
    min_point, _ = bounding_box
    return min_point[2]


def get_xmax(bounding_box):
    _, max_point = bounding_box
    return max_point[0]


def get_ymax(bounding_box):
    _, max_point = bounding_box
    return max_point[1]


def get_zmax(bounding_box):
    _, max_point = bounding_box
    return max_point[2]


def bottom_bounding_box_point(bounding_box, plane_normal):
    """
    From the eight corners of the bounding box, return the one that is the "lowest" with respect to the plane defined by the normal vector.
    If multiple points are equally low, return the one with the leftmost/frontmost coordinates according to these rules:
    - For normal (0,0,1): bottom leftmost/frontmost (lowest y)
    - For normal (0,0,-1): top leftmost/frontmost
    - For normal (1,1,1): bottom leftmost
    - For normal (-1,-1,-1): top rightmost (biggest x) back (biggest y)

    :param bounding_box: Tuple of two points defining the bounding box ((xmin, ymin, zmin), (xmax, ymax, zmax))
    :param plane_normal: Normal vector of the reference plane (e.g., (0, 0, 1) for XY plane)
    :return: The corner point as a tuple (x, y, z)
    """
    from itertools import product

    import numpy as np

    min_point, max_point = bounding_box

    # Generate all 8 corners using Cartesian product
    corners = list(
        product(
            [min_point[0], max_point[0]],  # x coordinates
            [min_point[1], max_point[1]],  # y coordinates
            [min_point[2], max_point[2]],  # z coordinates
        )
    )

    # Normalize the plane normal
    normal = np.array(plane_normal)
    normal = normal / np.linalg.norm(normal)

    # Calculate the dot product of each corner with the normal (distance along normal)
    distances = [np.dot(corner, normal) for corner in corners]

    # Find the minimum distance (lowest point along the normal direction)
    min_distance = min(distances)

    # Find all corners that have the minimum distance
    candidate_corners = [
        corners[i] for i, d in enumerate(distances) if abs(d - min_distance) < 1e-10
    ]

    # If only one corner, return it
    if len(candidate_corners) == 1:
        return candidate_corners[0]

    # Tie-breaking: choose based on the normal vector direction
    # The logic is to choose the corner that aligns with the "leftmost/frontmost" preference
    # For positive normal components, prefer smaller coordinates
    # For negative normal components, prefer larger coordinates

    def tie_breaker_score(corner):
        # Create a score where smaller is better for the preferred corner
        x, y, z = corner
        nx, ny, nz = normal

        # For each axis, if normal component is positive, prefer smaller coordinate
        # If normal component is negative, prefer larger coordinate
        score_x = x if nx >= 0 else -x
        score_y = y if ny >= 0 else -y
        score_z = z if nz >= 0 else -z

        # Return tuple for lexicographic ordering (x first, then y, then z)
        return (score_x, score_y, score_z)

    # Sort by the tie-breaker score and return the best one
    best_corner = min(candidate_corners, key=tie_breaker_score)
    return best_corner
