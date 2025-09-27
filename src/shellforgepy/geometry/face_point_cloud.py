import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

freecad_utils_path = "/Users/mege/git/mege_3d/freecad/mege_freecad_utils"

parent_dir = str(Path(freecad_utils_path).parent)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from shellforgepy.construct.construct_utils import fibonacci_sphere
from shellforgepy.geometry.spherical_tools import (
    cartesian_to_spherical_jackson,
    filter_outside_spherical_cap,
)

border_extension = 0.1

sphere_radius = 0.22


parameters_m = {
    "parameters_in_face_percent": {
        "nose_length": 17,
        "nose_height": 18,
        "nose_halfway_protrusion": 3,
        "nostril_y": -3.8,
        "nose_base_width_top": 8.1,
        "nose_base_top_protrusion": -2,
        "nostril_width": 10.5,
        "nostril_base_protrusion": 1.5,
        "nose_bottom_y": -6,
        "nose_back_width": 3,
        "nose_back_y_offset_bottom": 1.0,
        "nose_top_protrusion": 1.5,
        "eye_outer_point_distance": 45,
        "eye_distance": 28,
        "eye_outer_point_protrusion": 0,
        "eye_depth": -6,
        "eye_size": 4.8,
        "eye_y": 17,
        "chin_width": 10.1,
        "chin_center_y_offset": 2.9,
        "chin_protrusion": 10,
        "chin_bottom_protrusion": 12,
        "chin_y": -33.0,
        "chin_tip_height": 8.6,
        "chin_base_width": 16.2,
        "cheekbone_width": 44,
        "cheekbone_y": 6.0,
        "cheekbone_protrusion": 1,
        "lip_top_center_width": 9.5,
        "lip_protrusion": 1,
        "mouth_width": 24,
        "face_width_at_mouth": 34,
        "mouth_center_y": -18.0,
        "upper_lip_center_y__offset": -1.9,
        "lower_lip_center_y__offset": 0.0,
        "mouth_center_y_offset": -1.9,
        "smile_offset": 1,
        "lip_size": 3.7,
        "brow_width": 45,
        "brow_y": 36.3,
        "brow_top": 28,
        "brow_protrusion": 4,
        "brow_top_protrusion": 4.5,
        "jaw_protrusion": -5,
    },
    "brow_top_factor": 1.3,
    "aspect_ratio": 0.6,
    "x_squeeze": 0.65,
    "z_squeeze": 0.8,
}


parameters_n = {
    "parameters_in_face_percent": {
        "nose_length": 12,
        "nose_height": 18,
        "nose_halfway_protrusion": 3,
        "nostril_y": -3.8,
        "nose_base_width_top": 8.1,
        "nose_base_top_protrusion": -2,
        "nostril_width": 13,
        "nostril_base_protrusion": 1.5,
        "nose_bottom_y": -6,
        "nose_back_width": 3,
        "nose_back_y_offset_bottom": 5.0,
        "nose_top_protrusion": 1.5,
        "eye_outer_point_distance": 45,
        "eye_distance": 28,
        "eye_outer_point_protrusion": 0,
        "eye_depth": -6,
        "eye_size": 4.8,
        "eye_y": 17,
        "chin_width": 19,
        "chin_base_width": 15,
        "chin_center_y_offset": 2.9,
        "chin_protrusion": 8,
        "chin_bottom_protrusion": 12,
        "chin_y": -35.0,
        "chin_tip_height": 8.6,
        "cheekbone_width": 48,
        "cheekbone_y": 4.3,
        "cheekbone_protrusion": 1,
        "lip_top_center_width": 9.5,
        "lip_protrusion": 1,
        "mouth_width": 25,
        "face_width_at_mouth": 41,
        "mouth_center_y": -18.0,
        "upper_lip_center_y__offset": -1.9,
        "lower_lip_center_y__offset": 0.0,
        "mouth_center_y_offset": -1.9,
        "smile_offset": 1,
        "lip_size": 3.7,
        "brow_width": 45,
        "brow_y": 36.3,
        "brow_top": 28,
        "brow_protrusion": 4,
        "brow_top_protrusion": 4.5,
        "jaw_protrusion": -5,
    },
    "brow_top_factor": 1.0,
    "aspect_ratio": 0.8,
    "x_squeeze": 0.7,
    "z_squeeze": 0.8,
}


def inverse_azimuthal_projection(xy_plane: np.ndarray):
    x_, y_ = xy_plane.T
    theta = np.sqrt(x_**2 + y_**2)
    phi = np.arctan2(y_, x_)
    return np.stack([theta, phi], axis=1)


def azimuthal_to_cartesian(xy_plane: np.ndarray, delta_r: np.ndarray):
    theta_phi = inverse_azimuthal_projection(xy_plane)
    theta, phi = theta_phi.T
    r = 1.0 + delta_r
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=1)


def spherical_to_azimuthal_projection(theta, phi):
    # theta: angle from north pole (0 = north pole, π = south pole)
    # phi: azimuth (longitude)
    rho = theta
    x_ = rho * np.cos(phi)
    y_ = rho * np.sin(phi)
    return np.stack([x_, y_], axis=1)


def build_face_cap_grid_with_deltas(parameters_to_use, outer_deltar=0):

    face_overall_size = 1.4
    detar_scale = 1

    outer_deltar = outer_deltar * face_overall_size / 100
    parameters_in_face_percent = parameters_to_use["parameters_in_face_percent"]
    brow_top_factor = parameters_to_use["brow_top_factor"]
    aspect_ratio = parameters_to_use["aspect_ratio"]

    parameters_scaled = {
        k: v * face_overall_size / 100 for k, v in parameters_in_face_percent.items()
    }
    par = SimpleNamespace(**parameters_scaled)

    points = []

    def new_point_too_close(points, new_point, threshold=0.005):
        for i, point in enumerate(points):
            distance = np.linalg.norm(np.array(point[:2]) - np.array(new_point[:2]))
            if distance < threshold:
                print(f"Point {new_point} too close to existing point {i} :  {point}")
                return True
        return False

    labels = []

    def add_point(points, new_point, label=""):
        if not new_point_too_close(points, new_point):
            points.append(new_point)
            labels.append(label)
        else:
            raise ValueError("Point too close to existing points")

    rho = face_overall_size
    alphas = [np.radians(a) for a in [90, 135, 165, 225, 270]]
    for outer_point_number, alpha in enumerate(alphas):

        deltar = outer_deltar if np.degrees(alpha) < 200 else par.jaw_protrusion
        x_ = rho * np.cos(alpha) * aspect_ratio
        y_ = rho * np.sin(alpha)
        add_point(points, (x_, y_, deltar), f"outer_point_{outer_point_number}")
        if abs(x_) > 1e-3:
            add_point(
                points, (-x_, y_, deltar), f"outer_point_{outer_point_number}_neg"
            )

    add_point(points, (0, 0, par.nose_length), "nose_tip")
    add_point(points, (0, par.nose_bottom_y, 0), "nose_bottom")

    add_point(
        points, (0, par.mouth_center_y + par.mouth_center_y_offset, 0), "mouth_center"
    )
    add_point(
        points,
        (
            0,
            par.mouth_center_y + par.lip_size + par.upper_lip_center_y__offset,
            par.lip_protrusion,
        ),
        "upper_lip_center",
    )
    add_point(
        points,
        (
            0,
            par.mouth_center_y - par.lip_size + par.lower_lip_center_y__offset,
            par.lip_protrusion,
        ),
        "lower_lip_center",
    )
    add_point(
        points,
        (0, par.mouth_center_y - 2 * par.lip_size + par.lower_lip_center_y__offset, 0),
        "lower_lip_center",
    )

    add_point(
        points,
        (0, par.chin_y + par.chin_center_y_offset, par.chin_protrusion),
        "chin_center",
    )
    add_point(points, (0, par.brow_y, par.brow_protrusion), "brow_center")

    for i in [-1, 1]:
        add_point(
            points,
            (
                i * par.nose_base_width_top,
                par.nose_height,
                par.nose_base_top_protrusion,
            ),
            "nose_base_top",
        )
        add_point(
            points,
            (
                i * par.nose_back_width,
                par.nose_height / 2,
                par.nose_base_top_protrusion / 2
                + par.nose_length / 2
                + par.nose_halfway_protrusion,
            ),
            "nose_halfway",
        )

        add_point(
            points,
            (i * par.nostril_width, par.nostril_y, par.nostril_base_protrusion),
            "nostril_base",
        )
        add_point(
            points,
            (i * par.nose_back_width, par.nose_height, par.nose_top_protrusion),
            "nose_back_at_tip",
        )
        add_point(
            points,
            (i * par.nose_back_width, par.nose_back_y_offset_bottom, par.nose_length),
            "nose_back_at_bottom",
        )

        add_point(
            points,
            (
                i * par.nostril_width,
                par.nose_back_y_offset_bottom,
                par.nostril_base_protrusion,
            ),
            "nostril_base_at_bottom",
        )
        add_point(
            points,
            (
                i * (par.nose_base_width_top / 2 + par.nostril_width / 2),
                par.nose_height / 2,
                par.nostril_base_protrusion / 2 + par.nose_base_top_protrusion / 2,
            ),
            "nostril_base_halfway",
        )

        add_point(
            points,
            (
                i * par.lip_top_center_width,
                par.mouth_center_y + par.lip_size + par.upper_lip_center_y__offset,
                par.lip_protrusion,
            ),
            "upper_lip_top",
        )
        add_point(
            points,
            (i * par.lip_top_center_width, par.mouth_center_y, 0),
            "upper_lip_bottom",
        )
        add_point(
            points,
            (
                i * par.lip_top_center_width,
                par.mouth_center_y - par.lip_size + par.lower_lip_center_y__offset,
                par.lip_protrusion,
            ),
            "lower_lip_top",
        )
        add_point(
            points,
            (
                i * par.lip_top_center_width,
                par.mouth_center_y - 2 * par.lip_size + par.lower_lip_center_y__offset,
                par.lip_protrusion,
            ),
            "lower_lip_bottom",
        )
        add_point(
            points,
            (i * par.mouth_width, par.mouth_center_y + par.smile_offset, 0),
            "mouth_corner",
        )

        add_point(
            points,
            (i * par.face_width_at_mouth, par.mouth_center_y, 0),
            "face_outer_points_at_mouth",
        )

        add_point(
            points, (i * par.chin_width, par.chin_y, par.chin_protrusion), "chin_base"
        )
        add_point(
            points,
            (
                i * par.chin_base_width,
                par.chin_y - par.chin_tip_height,
                par.chin_bottom_protrusion,
            ),
            "chin_bottom",
        )

        brow_points = 3
        for j in range(1, brow_points):
            x_relative = j / (brow_points - 1) * par.brow_width
            add_point(
                points, (i * x_relative, par.brow_y, par.brow_protrusion), "brow_bottom"
            )
            add_point(
                points,
                (
                    i * x_relative * brow_top_factor,
                    par.brow_y + par.brow_top,
                    par.brow_top_protrusion,
                ),
                "brow_top",
            )

        add_point(
            points,
            (i * par.cheekbone_width, par.cheekbone_y, par.cheekbone_protrusion),
            "cheekbone",
        )
        add_point(
            points,
            (
                i * par.eye_outer_point_distance,
                par.eye_y,
                par.eye_outer_point_protrusion,
            ),
            "eye_outer_point",
        )
        add_point(points, (i * par.eye_distance, par.eye_y, par.eye_depth), "eye")

    xy_points = np.array([[x, y] for (x, y, _) in points])
    delta_r = np.array([z * detar_scale for (_, _, z) in points])
    return xy_points, delta_r, labels


def face_point_cloud(face_key):

    if face_key not in ["n", "m"]:
        raise ValueError("face_key must be 'n' or 'm'")
    if face_key == "n":
        parameters_to_use = parameters_n
    elif face_key == "m":
        parameters_to_use = parameters_m

    xy_plane_points, delta_r, labels = build_face_cap_grid_with_deltas(
        parameters_to_use=parameters_to_use, outer_deltar=0.1
    )
    points_3d = azimuthal_to_cartesian(xy_plane_points, delta_r) * sphere_radius

    spherical_coords = [cartesian_to_spherical_jackson(v) for v in points_3d]

    spherical_unit_sphere_coords = [
        (1, theta, phi) for r, theta, phi in spherical_coords
    ]
    cap_theta_phi = np.array([(rtp[1], rtp[2]) for rtp in spherical_unit_sphere_coords])

    n_theta = 24  # azimuth (around Z, like longitude)
    n_phi = int(n_theta * 0.4)  # inclination (from +Z pole to -Z)

    theta = np.linspace(-np.pi, np.pi, n_theta)
    phi = np.linspace(0, np.pi, n_phi)  # from top (0) to bottom (π)

    theta_grid, phi_grid = np.meshgrid(theta, phi)
    theta_flat = theta_grid.ravel()
    phi_flat = phi_grid.ravel()

    grid_on_unit_sphere = fibonacci_sphere(int(n_theta * n_phi / 3))

    grid_on_unit_sphere_spherical = [
        cartesian_to_spherical_jackson(v) for v in grid_on_unit_sphere
    ]
    grid_theta_phi = np.array(
        [(rtp[1], rtp[2]) for rtp in grid_on_unit_sphere_spherical]
    )

    outside_theta_phi, mask_outside, hull_vertices = filter_outside_spherical_cap(
        cap_theta_phi, grid_theta_phi, border_extension=border_extension
    )

    grid_on_unit_sphere_np = np.array(grid_on_unit_sphere)

    x_squeeze = parameters_to_use["x_squeeze"]
    z_squeeze = parameters_to_use["z_squeeze"]

    all_points = [p for p in points_3d]
    all_labels = [l for l in labels]
    for point in grid_on_unit_sphere_np[mask_outside]:
        all_points.append(
            (
                point[0] * sphere_radius,
                point[1] * sphere_radius,
                point[2] * sphere_radius,
            )
        )
        all_labels.append("grid point")

    all_points = np.array(all_points)

    all_points = all_points * np.array([x_squeeze, 1.0, z_squeeze])

    # find "top" (max y) and "bottom" (min y) point indices

    all_points_spherical = [cartesian_to_spherical_jackson(v) for v in all_points]

    top_index = np.argmin([point[1] for point in all_points_spherical])
    bottom_index = np.argmax([point[1] for point in all_points_spherical])

    top_index = np.argmax(all_points[:, 1])
    bottom_index = np.argmin(all_points[:, 1])

    back_index = np.argmin(all_points[:, 2])

    all_labels[top_index] = "top"
    all_labels[bottom_index] = "bottom"
    all_labels[back_index] = "back"

    return all_points, all_labels
