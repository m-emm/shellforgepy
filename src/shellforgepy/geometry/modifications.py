from itertools import product

import numpy as np
from shellforgepy.adapters._adapter import create_box, get_bounding_box, get_volume
from shellforgepy.construct.alignment_operations import rotate, translate
from shellforgepy.construct.bounding_box_helpers import bottom_bounding_box_point
from shellforgepy.construct.construct_utils import normalize
from shellforgepy.geometry.spherical_tools import (
    coordinate_system_transform,
    coordinate_system_transform_to_matrix,
    coordinate_system_transformation_function,
    matrix_to_coordinate_system_transformation_function,
)


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
