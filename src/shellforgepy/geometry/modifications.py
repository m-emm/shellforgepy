import logging
from itertools import product

import numpy as np
from shellforgepy.adapters._adapter import (
    create_box,
    get_bounding_box,
    get_bounding_box_size,
    get_volume,
)
from shellforgepy.construct.alignment_operations import rotate, translate
from shellforgepy.construct.bounding_box_helpers import bottom_bounding_box_point
from shellforgepy.construct.construct_utils import (
    fibonacci_sphere,
    normalize,
    rotation_matrix_from_vectors,
)
from shellforgepy.geometry.spherical_tools import (
    coordinate_system_transform,
    coordinate_system_transform_to_matrix,
    coordinate_system_transformation_function,
    matrix_to_coordinate_system_transformation_function,
)

_logger = logging.getLogger(__name__)


def slice_part(
    part,
    slice_plane_normal,
    slice_thickness,
    transform_to_horizontal=True,
    start_point=None,
    slicing_length=None,
):
    """
    Slice a part into multiple parts along a specified plane normal with given thickness.

    Parameters:
    -----------
    part : solid object
        The part to be sliced. Must be compatible with get_bounding_box and cut operations.
    slice_plane_normal : array-like, shape (3,)
        Normal vector defining the slicing direction. Will be normalized internally.
    slice_thickness : float
        Thickness of each slice in the direction of the normal vector.
    transform_to_horizontal : bool, optional
        If True (default), each slice is rotated and translated to be horizontal,
        centered about the origin, and with z_min at 0.
        If False, slices remain in their original positions in the global coordinate system.
    start_point : array-like, shape (3,), optional
        Custom starting point for slicing. If None, automatically determined from bounding box.
        This allows consistent slicing across multiple parts with the same segmentation.
    slicing_length : float, optional
        Total length to slice along the normal direction. If None, automatically determined
        from bounding box. Combined with start_point, this allows precise control over the
        slicing extent for consistent segmentation across multiple parts.

    Returns:
    --------
    list of dict
        Each dictionary contains:
        - 'part': the sliced part (transformed if transform_to_horizontal=True)
        - 'plane_point': the point on the slicing plane
        - 'height': the height offset from the starting position
        - 'slice_bbox': bounding box of the slice for convenience
    """

    def point_in_upper_half_space(point, plane_point, plane_normal):
        """
        Check if a point is in the upper half-space defined by a plane.

        Parameters:
        -----------
        point : array-like, shape (3,)
            The point to test
        plane_point : array-like, shape (3,)
            A point on the plane
        plane_normal : array-like, shape (3,)
            Normal vector of the plane (should be normalized)

        Returns:
        --------
        bool
            True if point is in the upper half-space (same side as normal direction)
        """

        point = np.asarray(point, dtype=np.float64)
        plane_point = np.asarray(plane_point, dtype=np.float64)
        plane_normal = np.asarray(plane_normal, dtype=np.float64)

        # Vector from plane point to test point
        to_point = point - plane_point

        # If dot product is positive or very close to zero, point is in upper half-space
        # Use small tolerance to handle floating-point precision issues
        return np.dot(to_point, plane_normal) >= -1e-10

    slice_plane_normal = np.array(slice_plane_normal, dtype=np.float64)
    slice_plane_normal = normalize(slice_plane_normal)

    bounding_box = get_bounding_box(part)
    min_point, max_point = bounding_box

    # Generate all 8 corners of the bounding box
    corners = list(
        product(
            [min_point[0], max_point[0]],  # x coordinates
            [min_point[1], max_point[1]],  # y coordinates
            [min_point[2], max_point[2]],  # z coordinates
        )
    )

    # Determine starting point: use custom start_point if provided, otherwise compute from bounding box
    if start_point is not None:
        current_slice_start = np.array(start_point, dtype=np.float64)
    else:
        current_slice_start = np.array(
            bottom_bounding_box_point(bounding_box, slice_plane_normal),
            dtype=np.float64,
        )

    # Determine slicing extent: use custom slicing_length if provided, otherwise compute from bounding box
    if slicing_length is not None:
        total_slicing_distance = float(slicing_length)
        # Calculate end point for termination condition
        end_point = current_slice_start + slice_plane_normal * total_slicing_distance
    else:
        # Use bounding box corners for termination condition (original behavior)
        end_point = None
        total_slicing_distance = None

    slices = []

    # Calculate the maximum diagonal length of the bounding box to ensure full coverage
    # This ensures the cutter box is large enough to cut through the entire part
    max_distance = 0
    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            dist = np.linalg.norm(np.array(corners[i]) - np.array(corners[j]))
            if dist > max_distance:
                max_distance = dist
    part_diagonal_length = max_distance * 2  # Add 100% margin for safety

    # Create cutter: a large box that extends infinitely in both directions along the normal
    # The gap between bottom_cutter and top_cutter defines the slice thickness
    bottom_cutter = create_box(
        part_diagonal_length, part_diagonal_length, part_diagonal_length
    )
    bottom_cutter = translate(
        -part_diagonal_length / 2, -part_diagonal_length / 2, -part_diagonal_length
    )(bottom_cutter)

    top_cutter = create_box(
        part_diagonal_length, part_diagonal_length, part_diagonal_length
    )
    top_cutter = translate(
        -part_diagonal_length / 2, -part_diagonal_length / 2, slice_thickness
    )(top_cutter)

    cutter = top_cutter.fuse(bottom_cutter)
    current_height = 0
    slice_count = 0  # Track actual number of slices created

    # Set up initial termination condition check
    if slicing_length is not None:
        # Custom slicing length: check distance from start point
        initial_check_passed = True  # Always proceed with custom length
    else:
        # Original behavior: verify that initially all corners are in the upper half-space
        # (i.e., we start from the correct side of the part)
        initial_plane_point = current_slice_start
        corners_in_upper_space = [
            point_in_upper_half_space(corner, initial_plane_point, slice_plane_normal)
            for corner in corners
        ]
        initial_check_passed = any(corners_in_upper_space)

    if not initial_check_passed:
        # If no corners are in upper half-space initially, we might have the normal backwards
        # or the starting point is wrong. This is a degenerate case.
        return slices

    while True:
        bottom_slice_plane_point = current_slice_start

        # Choose an appropriate "out" vector that's not parallel to slice_plane_normal
        # This is used as a reference direction for the coordinate transformation
        out = np.array([0, 0, 1], dtype=np.float64)
        if abs(np.dot(slice_plane_normal, out)) > 0.9:  # Nearly parallel
            out = np.array([1, 0, 0], dtype=np.float64)
            if abs(np.dot(slice_plane_normal, out)) > 0.9:  # Still nearly parallel
                out = np.array([0, 1, 0], dtype=np.float64)

        transform_function = coordinate_system_transformation_function(
            (0, 0, 0),  # source origin
            (0, 0, 1),  # source up (Z axis)
            (1, 0, 0),  # source out (X axis)
            bottom_slice_plane_point,  # target origin
            slice_plane_normal,  # target up (slice normal)
            out,  # target out
            rotate,  # rotation function generator (first)
            translate,  # translation function generator (second)
        )

        transformed_cutter = transform_function(cutter)
        sliced_part = part.cut(transformed_cutter)

        # Check if the sliced part has meaningful volume
        slice_volume = get_volume(sliced_part)
        if slice_volume > 1e-10:  # Only keep slices with non-zero volume

            if transform_to_horizontal:
                # Create inverse transformation to orient slice horizontally
                # We want to transform from the slice coordinate system back to a canonical orientation
                # Target: horizontal slice with z-axis as normal, centered and resting at z=0

                # Compute the transform from canonical orientation to slice orientation
                forward_transform = coordinate_system_transform(
                    origin_a=(0, 0, 0),  # canonical origin
                    up_a=(0, 0, 1),  # canonical up (Z axis)
                    out_a=(1, 0, 0),  # canonical out (X axis)
                    origin_b=bottom_slice_plane_point,  # slice origin
                    up_b=slice_plane_normal,  # slice up (normal)
                    out_b=out,  # slice out
                )

                # Convert to matrix and invert it to get slice-to-canonical transform
                forward_matrix = coordinate_system_transform_to_matrix(
                    forward_transform
                )
                inverse_matrix = np.linalg.inv(forward_matrix)

                # Create transformation function from the inverse matrix
                inverse_transform_function = (
                    matrix_to_coordinate_system_transformation_function(
                        inverse_matrix, rotate, translate
                    )
                )

                # Apply inverse transform to orient slice horizontally
                horizontal_slice = inverse_transform_function(sliced_part)

                # Get bounding box of the transformed slice
                slice_bbox = get_bounding_box(horizontal_slice)
                slice_min = np.array(slice_bbox[0])
                slice_max = np.array(slice_bbox[1])
                slice_center = (slice_min + slice_max) / 2

                # Center the slice at origin in X and Y, and place bottom at Z=0
                final_translation = translate(
                    -slice_center[0],  # center X
                    -slice_center[1],  # center Y
                    -slice_min[2],  # bottom at Z=0
                )

                final_slice = final_translation(horizontal_slice)
            else:
                # Keep slice in original position
                final_slice = sliced_part

            slices.append(
                {
                    "part": final_slice,
                    "plane_point": bottom_slice_plane_point.copy(),
                    "height": current_height,
                    "slice_index": slice_count,  # Add slice index for better tracking
                    "slice_bbox": get_bounding_box(final_slice),
                }
            )
            slice_count += 1

        # Advance to next slice position (always advance, regardless of volume)
        current_height += slice_thickness
        current_slice_start = current_slice_start + slice_plane_normal * slice_thickness

        # Determine termination condition based on whether custom slicing length is used
        if slicing_length is not None:
            # Custom slicing length: check if we've sliced the full distance
            if current_height >= total_slicing_distance:
                break
        else:
            # Original behavior: check if any corner of the bounding box is still in the upper half-space
            # If no corners remain in the upper half-space, we've sliced past the entire part
            next_plane_point = current_slice_start
            corners_still_in_upper_space = [
                point_in_upper_half_space(corner, next_plane_point, slice_plane_normal)
                for corner in corners
            ]

            if not any(corners_still_in_upper_space):
                # No more corners in upper half-space, we're done
                break

    return slices


def orient_for_flatness(part, samples=100, z_rotation_samples=8):
    """
    Orient a part to minimize its Z-height (flatten it as much as possible).

    This function uses Fibonacci sphere sampling to test different orientations
    and finds the one that results in the smallest Z-extent of the part's
    bounding box. For each sphere direction, it also tests different rotations
    around the Z-axis to find the globally optimal orientation.

    Parameters:
    -----------
    part : solid object
        The part to be oriented. Must be compatible with get_bounding_box_size
        and rotation operations.
    samples : int, optional
        Number of orientation samples to test using Fibonacci sphere
        distribution. More samples give better results but take longer.
        Default is 100.
    z_rotation_samples : int, optional
        Number of Z-axis rotation angles to test for each Fibonacci sphere
        direction. This captures the third rotational degree of freedom.
        Default is 8 (every 45 degrees).

    Returns:
    --------
    solid object
        The optimally oriented part with minimal Z-height.
    """
    # Get original Z-height
    original_size = get_bounding_box_size(part)
    original_z_height = original_size[2]

    if original_z_height < 1e-6:
        # Part is already flat or degenerate
        return part

    # Generate Fibonacci sphere directions to test
    directions = fibonacci_sphere(samples)

    # Generate Z-axis rotation angles to test (in degrees)
    z_angles = np.linspace(0, 360, z_rotation_samples, endpoint=False)

    best_z_height = original_z_height
    best_part = part

    for direction in directions:
        # Create rotation matrix to align this direction with +Z axis
        up_current = direction / np.linalg.norm(direction)
        up_target = np.array([0, 0, 1])

        # Use rotation_matrix_from_vectors to get the rotation
        R = rotation_matrix_from_vectors(up_current, up_target)

        # Convert to axis-angle representation for rotate_part
        # Extract axis and angle from rotation matrix
        trace = np.trace(R)
        angle_rad = np.arccos(np.clip((trace - 1) / 2, -1, 1))

        if angle_rad < 1e-6:
            # No rotation needed, direction is already aligned with Z
            primarily_oriented_part = part
        else:
            # Extract rotation axis
            if np.isclose(angle_rad, np.pi):
                # Special case for 180-degree rotation
                # Find eigenvector with eigenvalue 1
                eigenvals, eigenvecs = np.linalg.eigh(R)
                axis = eigenvecs[:, np.isclose(eigenvals, 1)].flatten()
            else:
                # General case
                axis = np.array(
                    [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
                ) / (2 * np.sin(angle_rad))

            axis = axis / np.linalg.norm(axis)
            angle_deg = np.degrees(angle_rad)

            # Apply primary rotation
            primarily_oriented_part = rotate(angle_deg, axis=tuple(axis))(part)

        # Now test different Z-axis rotations
        for z_angle in z_angles:
            if z_angle == 0:
                rotated_part = primarily_oriented_part
            else:
                # Apply Z-axis rotation
                rotated_part = rotate(z_angle, axis=(0, 0, 1))(primarily_oriented_part)

            # Get Z-height of rotated part
            size = get_bounding_box_size(rotated_part)
            z_height = size[2]

            # Keep track of best orientation
            if z_height < best_z_height:
                _logger.info(
                    f"New best Z-height: {z_height:.6f} (was {best_z_height:.6f})"
                )
                best_z_height = z_height
                best_part = rotated_part

    return best_part


# ---------------------------
# Geometry on the sphere S^2
# ---------------------------


def _unit(v):
    n = np.linalg.norm(v)
    return v if n == 0 else (v / n)


def _tangent_basis(u):
    """Return two orthonormal tangent vectors at unit u."""
    u = _unit(u)
    a = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = _unit(np.cross(u, a))
    t2 = np.cross(u, t1)  # already unit if u,t1 unit and orthogonal
    return t1, t2


def _exp_map(u, v_tan):
    """Exponential map at u: v_tan is tangent (3D), ||v_tan|| = step (radians)."""
    u = _unit(u)
    theta = np.linalg.norm(v_tan)
    if theta < 1e-15:
        return u
    k = v_tan / theta
    # exp_u(v) = cos(theta) u + sin(theta) k
    return _unit(np.cos(theta) * u + np.sin(theta) * k)


def _geodesic_point(u, dir_tan_unit, t):
    """Point at geodesic distance t from u along tangent unit dir."""
    return _exp_map(u, dir_tan_unit * t)


# ---------------------------
# Your limited API wrappers
# ---------------------------


def _shortest_arc_axis_angle(a, b):
    a = _unit(a)
    b = _unit(b)
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    if dot > 1 - 1e-12:
        return np.array([1, 0, 0]), 0.0
    if dot < -1 + 1e-12:
        axis = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
        axis = _unit(axis - a * np.dot(a, axis))
        return axis, 180.0
    axis = _unit(np.cross(a, b))
    ang = np.degrees(np.arccos(dot))
    return axis, ang


def _align_u_to_Z(part, u):
    """Rotate part so that direction u maps to +Z; return rotated part."""
    axis, ang = _shortest_arc_axis_angle(u, np.array([0.0, 0.0, 1.0]))
    if ang < 1e-12:
        return part
    return rotate(ang, axis=tuple(axis))(part)


class HeightOracle:
    """
    Caches evaluations h(u): Z-height after aligning u -> +Z.
    Uses only functional rotate + get_bounding_box_size.
    """

    def __init__(self, part, key_tol=1e-8):
        self.part = part
        self.memo = {}
        self.key_tol = key_tol
        self.num_calls = 0

    def _key(self, u):
        u = _unit(u)
        return tuple(np.round(u, 8))  # quantize for cache hits

    def h(self, u):
        k = self._key(u)
        if k in self.memo:
            return self.memo[k]
        rotated = _align_u_to_Z(self.part, u)
        z = float(get_bounding_box_size(rotated)[2])
        self.memo[k] = z
        self.num_calls += 1
        return z


# ---------------------------
# Derivative estimators
# ---------------------------


def _grad_central_diff(oracle, u, eps_rad=np.deg2rad(0.5)):
    """
    Central differences on the sphere in two tangent directions.
    Returns gradient vector in tangent space at u.
    """
    t1, t2 = _tangent_basis(u)

    u_p = _exp_map(u, t1 * eps_rad)
    u_m = _exp_map(u, -t1 * eps_rad)
    f_p = oracle.h(u_p)
    f_m = oracle.h(u_m)
    g1 = (f_p - f_m) / (2 * eps_rad)

    u_p = _exp_map(u, t2 * eps_rad)
    u_m = _exp_map(u, -t2 * eps_rad)
    f_p = oracle.h(u_p)
    f_m = oracle.h(u_m)
    g2 = (f_p - f_m) / (2 * eps_rad)

    g = g1 * t1 + g2 * t2  # tangent vector (not necessarily unit)
    return g


def _grad_spsa(oracle, u, c_rad=np.deg2rad(0.8), rng=None):
    """
    SPSA on S^2: two evaluations per iteration.
    Draw a random tangent direction v, estimate gradient along it, and
    map to a tangent vector. Noisy but cheap.
    """
    if rng is None:
        rng = np.random.default_rng()
    t1, t2 = _tangent_basis(u)
    xi1 = 1 if rng.random() < 0.5 else -1
    xi2 = 1 if rng.random() < 0.5 else -1
    v = _unit(xi1 * t1 + xi2 * t2)  # random tangent unit
    u_p = _exp_map(u, v * c_rad)
    u_m = _exp_map(u, -v * c_rad)
    f_p = oracle.h(u_p)
    f_m = oracle.h(u_m)
    df = (f_p - f_m) / (2 * c_rad)
    # Gradient estimate is df * v (in tangent); project scale via df
    return df * v


# ---------------------------
# Line search on a geodesic
# ---------------------------


def _golden_section_line_search(
    oracle, u, dir_tan, t_lo=0.0, t_hi=np.deg2rad(20.0), tol=np.deg2rad(0.1)
):
    """
    Minimize phi(t) = h(exp_u(t * dir_hat)) for t in [t_lo, t_hi].
    dir_tan can be any nonzero tangent vector.
    """
    dir_hat = _unit(dir_tan)
    phi = lambda t: oracle.h(_geodesic_point(u, dir_hat, t))

    gr = (np.sqrt(5) - 1) / 2  # ~0.618
    a, b = float(t_lo), float(t_hi)
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = phi(c)
    fd = phi(d)
    while (b - a) > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = phi(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = phi(d)
    t_star = 0.5 * (a + b)
    f_star = phi(t_star)
    return t_star, f_star


# ---------------------------
# Main optimizer
# ---------------------------


def orient_for_flatness_riemannian(
    part,
    coarse_samples=128,
    random_starts=2,
    max_iters=20,
    grad_method="central",  # "central" or "spsa"
    eps_rad=np.deg2rad(0.5),  # central diff step
    spsa_c_rad=np.deg2rad(0.8),  # SPSA perturbation
    line_search_hi=np.deg2rad(20.0),
    line_search_tol=np.deg2rad(0.1),
    improvement_tol=1e-4,  # mm threshold to stop
    seed=42,
    logger=None,
):
    """
    Returns: best_rotated_part, info_dict
    info: {'best_u','best_h','evals','starts','iterations','history':[(h,u),...]}
    """
    rng = np.random.default_rng(seed)
    oracle = HeightOracle(part)

    # Seed set: coarse Fibonacci + a few random
    def fib_sphere(n):
        i = np.arange(n, dtype=np.float64)
        phi = (1 + 5**0.5) / 2
        z = 1 - (2 * i + 1) / n
        r = np.sqrt(np.clip(1 - z * z, 0.0, 1.0))
        theta = 2 * np.pi * i / phi
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.stack([x, y, z], axis=1)

    seeds = list(fib_sphere(coarse_samples))
    for _ in range(random_starts):
        v = rng.normal(size=3)
        v = _unit(v)
        seeds.append(v)

    # Evaluate seeds and keep a handful of best as starts
    seed_vals = [(oracle.h(u), u) for u in seeds]
    seed_vals.sort(key=lambda p: p[0])
    starts = [u for (_, u) in seed_vals[: max(3, random_starts + 1)]]

    best_h, best_u = seed_vals[0]
    history = [(best_h, best_u)]

    # Choose gradient function
    if grad_method == "spsa":
        grad_fn = lambda u: _grad_spsa(oracle, u, c_rad=spsa_c_rad, rng=rng)
    else:
        grad_fn = lambda u: _grad_central_diff(oracle, u, eps_rad=eps_rad)

    total_iters = 0

    for u0 in starts:
        u = _unit(u0)
        h_curr = oracle.h(u)
        if logger:
            logger.info(f"[start] h={h_curr:.6f}  u={u}")

        for it in range(max_iters):
            total_iters += 1
            g = grad_fn(u)
            g_norm = np.linalg.norm(g)
            if g_norm < 1e-12:
                break
            # Descent direction in tangent
            d = -g

            # Geodesic line search
            t_star, h_next = _golden_section_line_search(
                oracle, u, d, t_lo=0.0, t_hi=line_search_hi, tol=line_search_tol
            )
            if h_curr - h_next < improvement_tol:
                # no meaningful improvement → stop this start
                break

            # Update along geodesic
            u = _geodesic_point(u, _unit(d), t_star)
            u = _unit(u)
            h_curr = h_next
            history.append((h_curr, u))
            if logger:
                logger.info(
                    f"  iter {it+1:02d}: h={h_curr:.6f}, step={np.degrees(t_star):.3f}°"
                )

        if h_curr < best_h:
            best_h, best_u = h_curr, u

    # Produce rotated part for the best direction
    rotated_best = _align_u_to_Z(part, best_u)
    info = {
        "best_u": best_u,
        "best_h": float(best_h),
        "evals": oracle.num_calls,
        "starts": len(starts),
        "iterations": total_iters,
        "history": history,
    }
    _logger.info(f"Optimization completed: {info}")
    return rotated_best
