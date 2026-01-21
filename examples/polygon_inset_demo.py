#!/usr/bin/env python3
"""
Demonstration of polygon inset functionality.

This script shows how to create inset polygons from various shapes,
demonstrating the `calc_inset_polygon_spec` method.
"""

import math

import numpy as np
from shellforgepy.construct.polygon_spec import PolygonSpec


def demo_triangle_inset():
    """Demonstrate inset on an equilateral triangle."""
    print("=== Triangle Inset Demo ===")

    # Create an equilateral triangle
    side_length = 2.0
    height = side_length * math.sqrt(3) / 2
    triangle = [
        np.array([0.0, 0.0, 0.0]),
        np.array([side_length, 0.0, 0.0]),
        np.array([side_length / 2, height, 0.0]),
    ]

    original = PolygonSpec(points=triangle)
    print(f"Original triangle perimeter: {original.circumference():.3f}")

    # Create inset versions
    inset_distances = [0.1, 0.2, 0.3]

    for inset_dist in inset_distances:
        try:
            inset_poly = original.calc_inset_polygon_spec(inset_dist)
            print(
                f"Inset by {inset_dist}: perimeter = {inset_poly.circumference():.3f}"
            )

            # Verify all vertices are inside
            all_inside = all(original.contains_point(v) for v in inset_poly.points)
            print(f"  All vertices inside original: {all_inside}")

        except ValueError as e:
            print(f"Inset by {inset_dist}: Failed - {e}")

    print()


def demo_square_inset():
    """Demonstrate inset on a square."""
    print("=== Square Inset Demo ===")

    # Create a square
    square = [
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        np.array([2.0, 2.0, 0.0]),
        np.array([0.0, 2.0, 0.0]),
    ]

    original = PolygonSpec(points=square)
    print(f"Original square perimeter: {original.circumference():.3f}")

    # Create inset versions
    inset_distances = [0.2, 0.4, 0.6, 0.8]

    for inset_dist in inset_distances:
        try:
            inset_poly = original.calc_inset_polygon_spec(inset_dist)
            print(
                f"Inset by {inset_dist}: perimeter = {inset_poly.circumference():.3f}"
            )

            # For a square, calculate expected side length
            expected_side = 2.0 - 2 * inset_dist
            actual_side = inset_poly.circumference() / 4
            print(f"  Expected side: {expected_side:.3f}, Actual: {actual_side:.3f}")

        except ValueError as e:
            print(f"Inset by {inset_dist}: Failed - {e}")

    print()


def demo_hexagon_inset():
    """Demonstrate inset on a regular hexagon."""
    print("=== Hexagon Inset Demo ===")

    # Create regular hexagon
    radius = 2.0
    hexagon = []
    for i in range(6):
        angle = i * 2 * math.pi / 6
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        hexagon.append(np.array([x, y, 0.0]))

    original = PolygonSpec(points=hexagon)
    print(f"Original hexagon perimeter: {original.circumference():.3f}")

    # Create inset versions
    inset_distances = [0.1, 0.3, 0.5, 0.7]

    for inset_dist in inset_distances:
        try:
            inset_poly = original.calc_inset_polygon_spec(inset_dist)
            print(
                f"Inset by {inset_dist}: perimeter = {inset_poly.circumference():.3f}"
            )

            # Check symmetry - all vertices should be equidistant from center
            center = inset_poly.center
            distances = [np.linalg.norm(v - center) for v in inset_poly.points]
            max_dist = max(distances)
            min_dist = min(distances)
            print(f"  Distance variation: {max_dist - min_dist:.6f} (should be ~0)")

        except ValueError as e:
            print(f"Inset by {inset_dist}: Failed - {e}")

    print()


def demo_tilted_polygon():
    """Demonstrate inset on a polygon in 3D space."""
    print("=== Tilted Polygon Demo ===")

    # Create a triangle in tilted plane
    triangle_3d = [
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 1.0, 5.0]),
        np.array([0.5, 5.0, 2.5]),
    ]

    original = PolygonSpec(points=triangle_3d)
    print(f"Original triangle normal: {original.normal}")
    print(f"Original triangle perimeter: {original.circumference():.3f}")

    # Create inset version
    inset_dist = 0.05
    try:
        inset_poly = original.calc_inset_polygon_spec(inset_dist)
        print(f"Inset triangle normal: {inset_poly.normal}")
        print(f"Inset triangle perimeter: {inset_poly.circumference():.3f}")

        # Check that normals are parallel
        dot_product = np.dot(original.normal, inset_poly.normal)
        print(f"Normal dot product: {dot_product:.6f} (should be ±1)")

        # Check all vertices are inside
        all_inside = all(original.contains_point(v) for v in inset_poly.points)
        print(f"All vertices inside original: {all_inside}")

    except ValueError as e:
        print(f"Inset failed: {e}")

    print()


def demo_multiple_iterations():
    """Demonstrate multiple inset iterations."""
    print("=== Multiple Iterations Demo ===")

    # Start with a large square
    square = [
        np.array([0.0, 0.0, 0.0]),
        np.array([5.0, 0.0, 0.0]),
        np.array([5.0, 5.0, 0.0]),
        np.array([0.0, 5.0, 0.0]),
    ]

    current_poly = PolygonSpec(points=square)
    inset_amount = 0.3

    print(f"Starting square perimeter: {current_poly.circumference():.3f}")

    for iteration in range(1, 8):
        try:
            current_poly = current_poly.calc_inset_polygon_spec(inset_amount)
            print(
                f"Iteration {iteration}: perimeter = {current_poly.circumference():.3f}"
            )

        except ValueError as e:
            print(f"Iteration {iteration}: Failed - {e}")
            break

    print()


if __name__ == "__main__":
    print("Polygon Inset Demonstration")
    print("=" * 40)
    print()

    demo_triangle_inset()
    demo_square_inset()
    demo_hexagon_inset()
    demo_tilted_polygon()
    demo_multiple_iterations()

    print("Demo completed!")
