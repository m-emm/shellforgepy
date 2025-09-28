import numpy as np
import pytest
from shellforgepy.geometry.face_point_cloud import (
    azimuthal_to_cartesian,
    build_face_cap_grid_with_deltas,
    face_point_cloud,
    inverse_azimuthal_projection,
    parameters_m,
    sphere_radius,
    spherical_to_azimuthal_projection,
)


def test_spherical_to_azimuthal_projection():
    """Test spherical to azimuthal projection conversion."""
    # Test basic projections with arrays
    thetas = np.array([0.5, 0.1, 1.0, 1.5])
    phis = np.array([1.0, 0.0, 0.5, 2.0])

    xy = spherical_to_azimuthal_projection(thetas, phis)

    # Check that result is finite and has correct shape
    assert isinstance(xy, np.ndarray)
    assert xy.shape == (4, 2)  # 4 points, 2 coordinates each
    assert np.all(np.isfinite(xy))

    # Test edge cases
    thetas_edge = np.array([0.0])
    phis_edge = np.array([0.0])
    xy_edge = spherical_to_azimuthal_projection(thetas_edge, phis_edge)
    assert np.all(np.isfinite(xy_edge))
    assert xy_edge.shape == (1, 2)


def test_inverse_azimuthal_projection():
    """Test inverse azimuthal projection."""
    # Test with multiple points
    xy_points = np.array([[0.0, 0.0], [0.1, 0.1], [0.5, 0.3], [-0.2, 0.4]])

    theta_phi = inverse_azimuthal_projection(xy_points)

    assert isinstance(theta_phi, np.ndarray)
    assert theta_phi.shape == (4, 2)  # 4 points, theta and phi for each
    assert np.all(np.isfinite(theta_phi))

    # Check that theta values are non-negative
    thetas = theta_phi[:, 0]
    assert np.all(thetas >= 0), "All theta values should be non-negative"

    # Test roundtrip consistency
    xy_back = spherical_to_azimuthal_projection(thetas, theta_phi[:, 1])
    assert np.allclose(xy_back, xy_points, atol=1e-6)


def test_azimuthal_to_cartesian():
    """Test azimuthal to cartesian conversion."""
    # Test basic conversion with multiple points
    xy_points = np.array([[0.1, 0.2], [0.0, 0.0], [0.5, 0.3]])
    delta_r = np.array([0.05, 0.03, 0.01])

    xyz = azimuthal_to_cartesian(xy_points, delta_r)

    assert isinstance(xyz, np.ndarray)
    assert xyz.shape == (3, 3)  # 3 points, 3 coordinates each
    assert np.all(np.isfinite(xyz))

    # Test with zero deltas
    delta_r_zero = np.array([0.0, 0.0, 0.0])
    xyz_zero = azimuthal_to_cartesian(xy_points, delta_r_zero)

    assert xyz_zero.shape == (3, 3)
    assert np.all(np.isfinite(xyz_zero))


def test_build_face_cap_grid_with_deltas():
    """Test face cap grid building with deltas."""
    # Use the actual parameter structure from parameters_n
    from shellforgepy.geometry.face_point_cloud import parameters_n

    # Test with default outer_deltar
    points, deltas, labels = build_face_cap_grid_with_deltas(parameters_n)

    assert isinstance(points, np.ndarray)
    assert isinstance(deltas, np.ndarray)
    assert isinstance(labels, list)
    assert points.ndim == 2
    assert deltas.ndim == 1  # 1D array of delta_r values
    assert points.shape[1] == 2  # 2D azimuthal projection points
    assert points.shape[0] == deltas.shape[0] == len(labels)
    assert np.all(np.isfinite(points))
    assert np.all(np.isfinite(deltas))

    # Test with non-zero outer_deltar
    points_outer, deltas_outer, labels_outer = build_face_cap_grid_with_deltas(
        parameters_n, outer_deltar=0.1
    )

    assert isinstance(points_outer, np.ndarray)
    assert isinstance(deltas_outer, np.ndarray)
    assert isinstance(labels_outer, list)
    assert points_outer.shape[0] == deltas_outer.shape[0] == len(labels_outer)
    assert np.all(np.isfinite(points_outer))
    assert np.all(np.isfinite(deltas_outer))


def test_face_point_cloud_basic_keys():
    """Test face_point_cloud function with valid face keys."""
    # Test with actually supported face keys
    face_keys = ["n", "m"]

    for face_key in face_keys:
        points, labels = face_point_cloud(face_key)

        assert isinstance(points, np.ndarray)
        assert isinstance(labels, list)
        assert points.ndim == 2
        assert points.shape[1] == 3  # 3D points
        assert points.shape[0] == len(labels)
        assert np.all(np.isfinite(points))


def test_face_point_cloud_invalid_key():
    """Test face_point_cloud with invalid key."""
    with pytest.raises((KeyError, ValueError, AttributeError)):
        face_point_cloud("invalid_face_key")


def test_parameters_m_structure():
    """Test that parameters_m has expected structure."""
    assert isinstance(parameters_m, dict)
    assert "parameters_in_face_percent" in parameters_m

    face_params = parameters_m["parameters_in_face_percent"]
    assert isinstance(face_params, dict)

    # Check some expected parameters exist
    expected_params = [
        "nose_length",
        "nose_height",
        "eye_distance",
        "chin_width",
        "eye_size",
        "eye_y",
    ]

    for param in expected_params:
        assert param in face_params
        assert isinstance(face_params[param], (int, float))


def test_sphere_radius_constant():
    """Test sphere_radius constant."""
    assert isinstance(sphere_radius, (int, float))
    assert sphere_radius > 0
    assert np.isfinite(sphere_radius)


def test_build_face_cap_grid_edge_cases():
    """Test edge cases for build_face_cap_grid_with_deltas."""
    from shellforgepy.geometry.face_point_cloud import parameters_m

    # Test with parameters_m (different from parameters_n)
    points, deltas, labels = build_face_cap_grid_with_deltas(parameters_m)
    assert isinstance(points, np.ndarray)
    assert isinstance(deltas, np.ndarray)
    assert isinstance(labels, list)

    # Test with empty parameters - should raise KeyError
    with pytest.raises((KeyError, AttributeError)):
        build_face_cap_grid_with_deltas({})


def test_azimuthal_cartesian_consistency():
    """Test consistency between azimuthal and cartesian conversions."""
    # Generate test points
    xy_points = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])

    delta_r = np.array([0.1, 0.05, 0.02])  # Same number as points

    # Convert all points at once
    xyz = azimuthal_to_cartesian(xy_points, delta_r)

    # Check basic properties
    assert xyz.shape == (3, 3)  # 3 points, 3 coordinates each
    assert np.all(np.isfinite(xyz))

    # Check that the transformation is reasonable
    # (points should be in reasonable range for face modeling)
    norms = np.linalg.norm(xyz, axis=1)
    assert np.all(norms < 10.0), "All points should be in reasonable scale"


def test_projection_symmetry():
    """Test symmetry properties of projections."""
    # Test that projection is consistent with itself
    thetas = np.array([0.5, 0.5, 0.5, 0.5, 0.1, 1.0, 1.5])
    phis = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 0.5, 1.0, 2.0])

    # Forward projection
    xy = spherical_to_azimuthal_projection(thetas, phis)

    # Inverse projection
    theta_phi_back = inverse_azimuthal_projection(xy)
    thetas_back = theta_phi_back[:, 0]
    phis_back = theta_phi_back[:, 1]

    # Check consistency
    assert np.allclose(thetas, thetas_back, atol=1e-6)

    # For phi, we need to handle 2π periodicity and special case at theta=0
    for i, (theta, phi, phi_back) in enumerate(zip(thetas, phis, phis_back)):
        if theta < 1e-6:  # at theta=0, phi is undefined
            continue
        phi_diff = abs(phi - phi_back)
        phi_diff_mod = min(phi_diff, abs(phi_diff - 2 * np.pi))
        assert phi_diff_mod < 1e-6, f"Phi mismatch at index {i}: {phi} vs {phi_back}"


def test_face_point_cloud_output_properties():
    """Test properties of face_point_cloud outputs."""
    # Test with a valid face key
    points, labels = face_point_cloud("n")

    # Check output properties
    assert points.shape[0] > 0  # Should have some points
    assert points.shape[0] == len(labels)

    # Check coordinate ranges are reasonable for face modeling
    x_range = np.ptp(points[:, 0])  # peak-to-peak in x
    y_range = np.ptp(points[:, 1])  # peak-to-peak in y
    z_range = np.ptp(points[:, 2])  # peak-to-peak in z

    # All ranges should be finite and positive
    assert x_range >= 0 and np.isfinite(x_range)
    assert y_range >= 0 and np.isfinite(y_range)
    assert z_range >= 0 and np.isfinite(z_range)

    # Labels should be strings
    assert all(isinstance(label, str) for label in labels)
