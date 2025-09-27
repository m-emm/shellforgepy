import math
import time

overall_start_time = time.time()

from itertools import combinations

import networkx as nx
import numpy as np

from shellforgepy.simple import (
    get_bounding_box_center,
    get_vertex_coordinates,
    get_z_min,
    fibonacci_sphere,
    normalize,
    face_point_cloud,
    MeshPartition,
    PartitionableSpheroidTriangleMesh,
    coordinate_system_transform,
    coordinate_system_transform,
    coordinate_system_transform_to_matrix,
    coordinate_system_transformation_function,
    TransformedRegionView,
    create_trapezoidal_snake_geometry,
    align,
    Alignment,
    create_basic_box,
    PartCollector,
    translate,
    LeaderFollowersCuttersPart,
    PartList,
    directed_cylinder_at,
    create_solid_from_traditional_face_vertex_maps
)


def init_freecad():
    pass


quality_speed = 55  # Increased for PLA capability
inner_speed = 85  # Higher speeds for PLA

quality_acceleration = 2500  # Higher for PLA
inner_acceleration = 8000  # Higher for PLA

quality_jerk = 6  # Slightly higher for PLA
inner_jerk = 12  # Higher for PLA
nozzle_diameter = 0.6

min_layer_height = nozzle_diameter * 0.25
max_layer_height = nozzle_diameter * 0.75  # Slightly lower ratio for .6mm

##########
layer_height_factor = 0.6
##########

vertical_layers = 2

layer_height = np.round(
    min_layer_height + (max_layer_height - min_layer_height) * layer_height_factor, 2
)
print(
    f"Layer height: {layer_height} (nozzle_diameter: {nozzle_diameter}, min_layer_height: {min_layer_height}, max_layer_height: {max_layer_height})"
)


PROCESS_DATA_06_PLA = {
    "filament": "FilamentCrealityPLAHighSpeedTunedForSpeed",
    "process_overrides": {
        ######### .6MM NOZZLE SETTINGS #########
        "nozzle_diameter": "0.6",
        "max_layer_height": f"{max_layer_height}",
        "min_layer_height": f"{min_layer_height}",
        "layer_height": f"{layer_height}",
        "line_width": "0.65",  # ~108% of nozzle diameter for good flow
        "inner_wall_line_width": "0.65",
        "outer_wall_line_width": "0.6",  # Slightly smaller for quality
        "sparse_infill_line_width": "0.7",  # Wider for faster infill
        "initial_layer_line_width": "0.7",
        "solid_infill_line_width": "0.65",
        "support_interface_line_width": "0.6",
        "internal_solid_infill_line_width": "0.65",
        "support_line_width": "0.65",
        "bridge_line_width": "0.6",
        "thin_wall_line_width": "0.6",
        "gap_fill_line_width": "0.6",
        "top_surface_line_width": "0.6",  # Finer for surface quality
        ##### END .6MM NOZZLE SETTINGS #####
        # Basic setup
        "adaptive_layer_height": "0",
        "enable_arc_fitting": "1",
        # Layer and shell structure - adjusted for PLA strength requirements
        "bottom_shell_layers": "3",  # More layers for strength
        "top_shell_layers": "3",
        "wall_loops": "2",  # Keep 2 walls for strength
        "initial_layer_print_height": "0.3",
        # Infill - balanced for PLA
        "sparse_infill_density": "80%",
        "sparse_infill_pattern": "cubic",
        # Temperature settings for PLA
        "nozzle_temperature": "230",
        "nozzle_temperature_initial_layer": "235",  # Slightly higher for first layer
        "hot_plate_temp_initial_layer": "65",  # PLA bed temp
        # Cooling - PLA needs more cooling
        "fan_cooling_layer_time": "30",  # Shorter than PETG
        "fan_max_speed": "100",  # Full cooling for PLA
        "fan_min_speed": "80",  # High minimum for PLA
        "overhang_fan_speed": "100",
        "slow_down_for_layer_cooling": "1",
        "min_layer_time": "8",  # Shorter for PLA
        # Speed settings - leveraging PLA's printability
        "external_perimeter_speed": f"{quality_speed}",
        "initial_layer_infill_speed": f"{quality_speed}",
        "initial_layer_speed": f"{quality_speed}",
        "inner_wall_speed": f"{inner_speed}",
        "internal_solid_infill_speed": f"{inner_speed}",
        "gap_fill_speed": f"{inner_speed}",
        "gap_infill_speed": f"{inner_speed}",
        "solid_infill_speed": f"{inner_speed}",
        "sparse_infill_speed": f"{inner_speed}",
        "support_interface_speed": f"{inner_speed}",
        "support_speed": f"{inner_speed}",
        "top_surface_speed": f"{quality_speed}",
        "outer_wall_speed": f"{quality_speed}",
        "bridge_speed": "30",  # Faster bridges for PLA
        # Acceleration settings
        "initial_layer_acceleration": f"{quality_acceleration}",
        "outer_wall_acceleration": f"{quality_acceleration}",
        "top_surface_acceleration": f"{quality_acceleration}",
        "inner_wall_acceleration": f"{inner_acceleration}",
        "solid_infill_acceleration": f"{inner_acceleration}",
        "sparse_infill_acceleration": f"{inner_acceleration}",
        "support_acceleration": f"{inner_acceleration}",
        "support_interface_acceleration": f"{inner_acceleration}",
        "gap_fill_acceleration": f"{inner_acceleration}",
        "bridge_acceleration": f"{inner_acceleration}",
        # Jerk settings
        "initial_layer_jerk": f"{quality_jerk}",
        "outer_wall_jerk": f"{quality_jerk}",
        "top_surface_jerk": f"{quality_jerk}",
        "inner_wall_jerk": f"{inner_jerk}",
        "solid_infill_jerk": f"{inner_jerk}",
        "sparse_infill_jerk": f"{inner_jerk}",
        "support_interface_jerk": f"{inner_jerk}",
        "support_jerk": f"{inner_jerk}",
        "gap_fill_jerk": f"{inner_jerk}",
        # Retraction - PLA typically needs less retraction than PETG
        "filament_retraction_length": "1.2",  # Shorter for PLA
        "filament_retraction_speed": "40",  # Faster retraction
        "filament_deretraction_speed": "30",
        "filament_flow_ratio": "1.0",
        # Support settings - from PLA profiles
        "enable_support": "0",
        "bridge_no_support": "1",
        "support_threshold_angle": "50",  # Standard PLA support angle
        # Brim settings - from PLA profiles
        "brim_type": "outer_and_inner",  # Better adhesion than no_brim
        "brim_width": "4",
        "brim_ears_detection_length": "1",
        "brim_ears_max_angle": "125",
        "brim_object_gap": "0",
        # Surface quality and compensation
        "elefant_foot_compensation": "0.1",  # Less than PETG
        "xy_contour_compensation": "0",
        "xy_hole_compensation": "0.05",  # Less than PETG
        "infill_wall_overlap": "25%",  # Standard for PLA
        "resolution": "0.05",  # Finer resolution for PLA
    },
}


PROCESS_DATA = PROCESS_DATA_06_PLA

REALLY_DO_PROD = False


PROD = False

CREATE_SUPPORTS = False
LAY_FLAT_FOR_PROD = False
CREATE_EMBELLISHMENTS = False  # True
DRILL_AIR_HOLES = True

SHOW_REGION_IDS = False

REGION_TO_PRINT_LABEL = None  #  "nose_tip" # "back"  # None  #   "nose_tip"

REGION_TO_RRINT_ID = None  #  0  #  4  #  5  # None  # 5

NEVER_PRINT_REGION_IDS = [1, 8, 9]

NO_SUPPORTS_REGION_IDS = [0, 2, 6]

if REGION_TO_RRINT_ID is not None and REGION_TO_RRINT_ID in NO_SUPPORTS_REGION_IDS:
    print(
        f"Disabling supports for region {REGION_TO_RRINT_ID} as it is in the no-supports list."
    )
    PROCESS_DATA["process_overrides"]["enable_support"] = "0"
    PROCESS_DATA["process_overrides"]["brim_type"] = "outer_and_inner"
else:
    print(
        f"Enabling supports for region {REGION_TO_RRINT_ID} as it is not in the no-supports list."
    )
    PROCESS_DATA["process_overrides"]["enable_support"] = "1"
    PROCESS_DATA["process_overrides"]["brim_type"] = "no_brim"


if REALLY_DO_PROD:
    PROD = True
    LAY_FLAT_FOR_PROD = True
    SHOW_REGION_IDS = False
    DRILL_AIR_HOLES = True
else:
    REGION_TO_RRINT_ID = None


bottom_shaver_thickness = 0
embellishment_bottom_shaver_thickness = 0.5

CUTOUT_VIEW = False


CUTOUT_ANGLE = 90
CUTOUT_ANGLE_OFFSET = 90


EXPLOSION = None


if PROD:
    CUTOUT_VIEW = False

BIG_THING = 500

PROD_GAP = 3

BED_WIDTH = 220
MAX_BUILD_HEIGHT = 245


sphere_radius = 90
shell_thickness = 3

edge_marker_size = 0.7  # sphere_radius * 0.015

tile_shrinkage = 0.06
shrink_border = 0
edge_width = 10


base_radius = 9
base_height = 6

eye_hole_radius = sphere_radius * 0.18

brim_thickness = 0.3
brim_size = 2

connector_length = sphere_radius * 0.1
connector_thickness = sphere_radius * 0.01

connector_cylinder_length = connector_length * 1.05

connector_width = connector_length * 0.3
connector_cylinder_radius = connector_thickness * 0.8
connector_slack = 0.1


fall_support_interval = 5
fall_support_interval_growth = 1.5

fall_support_thickness = 0.2
fall_support_needle_size = fall_support_thickness * 0.4
fall_support_distance = 4
fall_support_triangle_gap = 0.1
fall_support_base_thickness = 0.4

tooth_depth = 4
tooth_width = 2.5


support_base_height = 8

rod_radius = 2.5

rod_tip_radius = 1
rod_tip_length = 9

allowed_unsupported_height = 60
required_distance = 10
min_cylinder_radius = 18
support_cylinder_shell_thickness = 0.5

supports_top_border = 20
num_support_sphere_points = 38

mege_head_diameter_side = 250
mege_head_diameter_up = 140
mege_head_diameter_front = 240

air_hole_diameter = 3
num_air_holes = 200  # 1500
min_air_hole_edge_distance = air_hole_diameter * 0.8

led_strip_width = 12
led_strip_thickness = 4.2
led_strip_length = 100  # 1000 / 3


coil_angle = -10


coil_start_angle_offset_degrees = -30
coil_y_offset = 84

coil_pitch = 18
coil_turns = 5

led_band_head_gap = 2.5
led_band_width = 14.38
led_band_thickness = 4.3
led_band_length = 2000
resolution = 60
marker_sphere_radius = 3

led_band_60pm_width = 12.1
led_band_60pm_thickness = 4.15

holder_thickness = 2
holder_horizontal_gap = 0.3
holder_vertical_gap = 0.3
holder_clamp_size = 2

holder_length = 8

holder_screw_size = "M3"

holder_screw_length = 10

holder_cutter_clearance = 0.4


def create_crude_led_band():
    retval = create_basic_box(
        led_band_width,
        led_band_length,
        led_band_thickness,
        origin=(-led_band_width / 2, 0, 0),
    )
    return retval


def create_air_hole_drills(region_view):
    air_hole_drills = PartCollector()
    print(f"Drilling air holes in region {region_view.region_id}")
    air_hole_sphere_points = fibonacci_sphere(num_air_holes)

    mesh_centroid = np.mean(region_view.partition.mesh.vertices, axis=0)

    transformed_mesh_centroid = region_view.transform_point(mesh_centroid)
    V, F, E = region_view.get_transformed_vertices_faces_boundary_edges()

    for direction_num, direction in enumerate(air_hole_sphere_points):

        if direction_num > 0 and direction_num % 100 == 0:
            print(
                f"Processed {direction_num} /  {len(air_hole_sphere_points)} air holes"
            )

        hits = region_view.ray_intersect_faces(transformed_mesh_centroid, direction)
        hit_face_ids = set()

        if hits is not None and len(hits) > 0:

            hit = hits[0]
            hit_point = hit[1]
            hit_face_id = hit[0]

            hit_face_ids.add(hit_face_id)

            # calculate minimal distance from hit point to any face edge

            min_edge_distance = np.inf
            for edge_number in range(3):
                edge_start_index = F[hit_face_id][edge_number]
                edge_end_index = F[hit_face_id][(edge_number + 1) % 3]
                edge_start = V[edge_start_index]
                edge_end = V[edge_end_index]
                edge_vector = edge_end - edge_start
                edge_vector = normalize(edge_vector)
                # project hit point onto edge vector
                hit_to_edge_start = hit_point - edge_start
                projection_length = np.dot(hit_to_edge_start, edge_vector)
                projection = edge_start + projection_length * edge_vector
                distance_to_edge = np.linalg.norm(hit_point - projection)
                if distance_to_edge < min_edge_distance:
                    min_edge_distance = distance_to_edge

            if min_edge_distance < min_air_hole_edge_distance:
                continue

            face_normal = region_view.face_normal(hit_face_id)
            drill_height = shell_thickness * 5

            drill_start = hit_point - np.array(direction) * (drill_height / 2)

            air_hole_drill = directed_cylinder_at(
                drill_start, face_normal, air_hole_diameter / 2, drill_height
            )

            air_hole_drills = air_hole_drills.fuse(air_hole_drill)

    return air_hole_drills


def create_partition():
    points_all_around, points_all_around_labels = face_point_cloud("m")

    points_all_around *= 1000 * sphere_radius / 100

    points = points_all_around

    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(
        points, vertex_labels=points_all_around_labels
    )

    partition = MeshPartition(mesh)

    min_z = np.min(mesh.vertices[:, 2])
    max_z = np.max(mesh.vertices[:, 2])

    chin_bottom_vertex = mesh.vertices[mesh.get_vertices_by_label("chin_bottom")[0]]

    partition = partition.perforate_and_split_region_by_plane(
        region_id=0,
        plane_point=chin_bottom_vertex,
        plane_normal=np.array([0, 1, 0.2]),
    )

    brow_cut_offset = 10
    brow_center_vertex = mesh.vertices[mesh.get_vertices_by_label("brow_center")[0]]

    cut_point_for_brow = brow_center_vertex + np.array([0, brow_cut_offset, 0])

    partition = partition.perforate_and_split_region_by_plane(
        region_id=0,
        plane_point=cut_point_for_brow,
        plane_normal=np.array([0, 1, 0]),
    )

    nose_tip_region_id = partition.find_regions_of_vertex_by_label("nose_tip")[0]

    partition = partition.perforate_and_split_region_by_plane(
        region_id=nose_tip_region_id,
        plane_point=np.array([0, 0, min_z + (max_z - min_z) * 0.65]),
        plane_normal=np.array([0, 0, 1]),
    )

    top_region_id = partition.find_regions_of_vertex_by_label("top")[0]

    top_region_vertices = np.array(
        list(partition.get_submesh_maps(top_region_id)["vertexes"].values())
    )

    max_z = np.max(top_region_vertices, axis=0)[2]
    min_z = np.min(top_region_vertices, axis=0)[2]

    cut_fraction = 0.5

    cut_z = min_z + (max_z - min_z) * cut_fraction

    partition = partition.perforate_and_split_region_by_plane(
        region_id=top_region_id,
        plane_point=(0, cut_z, 0),
        plane_normal=np.array([0, 0, 1]),
    )

    back_region_id = partition.find_regions_of_vertex_by_label("back")[0]

    back_region_vertices = partition.get_region_vertices(back_region_id)

    max_z_of_back = np.max(back_region_vertices[:, 2])

    cut_angle = np.degrees(66)

    cut_plane_normal_1 = np.array([np.sin(cut_angle), 0, np.cos(cut_angle)])

    partition = partition.perforate_and_split_region_by_plane(
        back_region_id,
        plane_point=(0, 0, max_z_of_back + 1),
        plane_normal=cut_plane_normal_1,
    )

    cut_plane_normal_2 = np.array([-np.sin(cut_angle), 0, np.cos(cut_angle)])
    back_region_id = partition.find_regions_of_vertex_by_label("back")[0]
    back_region_vertices = partition.get_region_vertices(back_region_id)

    max_z_of_back = np.max(back_region_vertices[:, 2])

    partition = partition.perforate_and_split_region_by_plane(
        back_region_id,
        plane_point=(0, 0, max_z_of_back + 1),
        plane_normal=cut_plane_normal_2,
    )

    back_region_id = partition.find_regions_of_vertex_by_label("back")[0]

    back_vertex = mesh.vertices[mesh.get_vertices_by_label("back")[0]]
    partition = partition.perforate_and_split_region_by_plane(
        region_id=back_region_id,
        plane_point=back_vertex,
        plane_normal=np.array([0, 1, 0]),
    )

    partition = partition.drill_holes_by_label(
        center_vertex_label="eye",
        radius=eye_hole_radius,
        height=sphere_radius,
        min_relative_area=1e-6,
        min_angle_deg=0.1,
    )

    print(f"Partitioned into {partition.get_regions()}")

    return partition


def create_embellishments(region_view):
    features_list_by_edge = {}
    partition = region_view.partition
    start_time = time.time()
    for edge in partition.mesh.get_canonical_edges():
        features = region_view.find_transformed_edge_features_along_original_edge(
            v0=edge[0],
            v1=edge[1],
        )
        features_list_by_edge[edge] = features

    elapsed_time = time.time() - start_time
    print(
        f"Find transformed edge features took {elapsed_time*1000:.0f} ms, found {len(features_list_by_edge)} edges with features"
    )

    start_time = time.time()

    embellishment_box_descriptors = []

    for edge, features in features_list_by_edge.items():
        start_time = time.time()
        for feature in features:
            p1, p2 = feature.edge_coords
            edge_vec = feature.edge_vector
            center = feature.edge_centroid

            # Create marker box: size along X

            # Define source and target coordinate systems
            origin_a = (0, 0, 0)
            up_a = (1, 0, 0)  # Local +X is box length
            out_a = (0, 0, 1)  # Local +Z is thickness

            origin_b = tuple(center)
            up_b = tuple(edge_vec)
            out_b = tuple(feature.face_normals[0])  # One of the adjacent face normals

            transformation = coordinate_system_transform(
                origin_a, up_a, out_a, origin_b, up_b, out_b
            )
            embellishment_box_descriptors.append((feature, transformation))
    elapsed_time = time.time() - start_time
    print(f"Calculating embellishment boxes took {elapsed_time*1000:.0f} ms")

    start_time = time.time()

    bed_cutter = create_basic_box(
        BIG_THING,
        BIG_THING,
        BIG_THING,
        origin=(-BIG_THING / 2, -BIG_THING / 2, -BIG_THING),
    )

    feature_collector = PartCollector()
    for j, (feature, transformation) in enumerate(embellishment_box_descriptors):
        if j % 10 == 0 and j > 0:
            elapsed_time = time.time() - start_time
            print(
                f"Materializized {j} / {len(embellishment_box_descriptors)} embellishment boxes, took {elapsed_time*1000:.0f} ms so far, {elapsed_time*1000/j:.0f} ms per box"
            )
        p1, p2 = feature.edge_coords
        edge_vec = feature.edge_vector
        length = np.linalg.norm(p2 - p1)

        box = create_basic_box(
            length,
            edge_marker_size,
            edge_marker_size,
            origin=(-length / 2, -edge_marker_size / 2, -edge_marker_size / 2),
        )
        box = rotate(45, axis=(1, 0, 0), center=(0, 0, 0))(box)

        rot = rotate(
            np.degrees(transformation["rotation_angle"]),
            axis=Base.Vector(*transformation["rotation_axis"]),
        )
        trans = translate(*transformation["translation"])

        box = rot(box)
        box = trans(box)

        feature_collector = feature_collector.fuse(box)
    elapsed_time = time.time() - start_time

    print(
        f"Materializing {len(embellishment_box_descriptors)} embellishment boxes took {elapsed_time*1000:.0f} ms"
    )
    return feature_collector


def calc_region_ids_to_print(partition, eye_regions):
    if REGION_TO_PRINT_LABEL is not None:

        region_to_print_id = partition.find_regions_of_vertex_by_label(
            REGION_TO_PRINT_LABEL
        )[0]
    elif REGION_TO_RRINT_ID is not None:
        region_to_print_id = REGION_TO_RRINT_ID
    else:
        region_to_print_id = None

    if region_to_print_id is not None and region_to_print_id in eye_regions:
        raise ValueError(
            f"Region to print {region_to_print_id} is an eye region, which is not allowed."
        )

    region_ids_to_print = set()
    for region_id in partition.get_regions():
        if (
            region_id != region_to_print_id and not region_to_print_id is None
        ) or region_id in NEVER_PRINT_REGION_IDS:
            continue
        region_ids_to_print.add(region_id)
    return region_ids_to_print


def calc_connector_region_id_pairs(partition, region_views, eye_regions):
    connector_region_id_pairs = combinations(
        [region_view.region_id for region_view in region_views], 2
    )

    connector_region_id_pairs = set(
        [tuple(sorted(pair)) for pair in connector_region_id_pairs]
    )

    bottom_region = partition.find_regions_of_vertex_by_label("bottom")[0]

    connector_region_id_pairs = set(
        [
            region_id_pair
            for region_id_pair in connector_region_id_pairs
            if bottom_region not in region_id_pair
        ]
    )
    connector_region_id_pairs = set(
        [
            region_id_pair
            for region_id_pair in connector_region_id_pairs
            if not (
                region_id_pair[0] in eye_regions or region_id_pair[1] in eye_regions
            )
        ]
    )

    return connector_region_id_pairs


def create_region_id_marker(region_view):
    region_id_text_object = create_text_object(
        str(region_view.region_id), size=10, thickness=1
    )
    bound_box_center = get_bounding_box_center(region_id_text_object)
    region_id_text_object = translate(*-bound_box_center)(region_id_text_object)

    V, F, _ = region_view.get_transformed_vertices_faces_boundary_edges()

    normals_sum = np.zeros(3)
    for i in range(len(F)):
        region_view.face_normal(i)
        normals_sum += region_view.face_normal(i)
    average_normal = normalize(normals_sum)

    centroid = np.mean(V, axis=0)

    vertex_nearest_to_centroid = np.argmin(np.linalg.norm(V - centroid, axis=1))
    centroid = V[vertex_nearest_to_centroid]

    pos = centroid + average_normal * 10

    region_id_text_object = translate(*pos)(region_id_text_object)

    return region_id_text_object


def lay_flat_for_prod(region_view, region_views):
    desired_region_pairs = set(
        combinations([region_view.region_id for region_view in region_views], 2)
    )

    undesired_region_parirs = set(
        [
            tuple(sorted(pair))
            for pair in [(1, 5), (0, 5), (4, 5), (0, 3), (3, 4), (1, 3)]
        ]
    )

    desired_region_pairs = desired_region_pairs - undesired_region_parirs

    region_view_edge_optimized = (
        region_view.lay_flat_on_boundary_edges_for_printability(
            desired_region_pairs=desired_region_pairs,
        )
    )

    region_view_flat_optimized = region_view.lay_flat_optimally_printable()

    score_edge_optimized = region_view_edge_optimized.printability_score()
    score_flat_optimized = region_view_flat_optimized.printability_score()

    if score_edge_optimized > score_flat_optimized:
        retval = region_view_edge_optimized
        print(
            f"Region {region_view.region_id} is edge-optimized with {score_edge_optimized} printability score."
        )
    else:
        retval = region_view_flat_optimized
        print(
            f"Region {region_view.region_id} is flat-optimized with {score_flat_optimized} printability score."
        )

    return retval


def create_connectors_and_connector_cutters(
    connector_hints, region_view, connector_region_id_pairs
):

    V, F, E = region_view.get_transformed_vertices_faces_boundary_edges()

    def male_female_region_calculator(hint):
        region_a, region_b = hint.region_a, hint.region_b
        # if one of the regions is 2, we define it to be the female region
        if region_a == 2:
            return region_b, region_a
        elif region_b == 2:
            return region_a, region_b

        elif region_a == 5:
            return region_b, region_a
        elif region_b == 5:
            return region_a, region_b
        else:
            male_region = max(region_a, region_b)
            female_region = min(region_a, region_b)

        return male_region, female_region

    connectors = PartCollector()
    cutters = PartCollector()
    region_id = region_view.region_id

    for hint in connector_hints:
        if region_id == hint.region_a or region_id == hint.region_b:

            materialized_connector = create_screw_connector_normal(
                hint,
                "M3",
                10,
                male_female_region_calculator=male_female_region_calculator,
            )

            edge_lenth = np.linalg.norm(hint.start_vertex - hint.end_vertex)
            if edge_lenth < connector_length * 3 and False:
                continue

            close_vertex_indices = set()

            for reference in [
                hint.start_vertex,
                hint.end_vertex,
                hint.edge_centroid,
            ]:
                close_vertex_indices.update(
                    region_view.vertex_indices_closer_than(
                        reference, 4 * connector_length
                    )
                )

            close_face_indices = region_view.face_indices_of_vertex_index_set(
                close_vertex_indices
            )
            for face_index in close_face_indices:

                face_vertices = region_view.face_vertices(face_index)

                bottom = np.array(
                    [face_vertices[0], face_vertices[1], face_vertices[2]]
                )

                # strech the bottom face by 1% around the centroid
                bottom_centroid = np.mean(bottom, axis=0)
                bottom = bottom_centroid + (bottom - bottom_centroid) * 1.01
                vertex_indices = F[face_index]

                top = np.zeros(shape=(3, 3))
                for i in range(3):
                    average_normal = region_view.average_normal_at_vertex(
                        vertex_indices[i]
                    )
                    top[i] = bottom[i] + 8 * connector_thickness * average_normal

                # Define vertex indices
                prism_map = {
                    "vertexes": {
                        0: top[0],
                        1: top[1],
                        2: top[2],
                        3: bottom[0],
                        4: bottom[1],
                        5: bottom[2],
                    },
                    "faces": {
                        0: [0, 2, 1],
                        1: [3, 4, 5],
                        2: [0, 1, 4],  # sides
                        3: [0, 4, 3],
                        4: [1, 2, 5],
                        5: [1, 5, 4],
                        6: [2, 0, 3],
                        7: [2, 3, 5],
                    },
                }
                cutter = create_solid_from_traditional_face_vertex_maps(prism_map)

                materialized_connector.male_connector = (
                    materialized_connector.male_connector.cut(cutter)
                )
                materialized_connector.female_connector = (
                    materialized_connector.female_connector.cut(cutter)
                )

            region_id_pair = tuple(
                sorted(
                    (
                        materialized_connector.male_region,
                        materialized_connector.female_region,
                    )
                )
            )

            if region_id_pair in connector_region_id_pairs:

                if region_id == materialized_connector.male_region:
                    if materialized_connector.male_connector is not None:
                        connectors = connectors.fuse(
                            materialized_connector.male_connector
                        )
                    if materialized_connector.male_cutter is not None:
                        cutters = cutters.fuse(materialized_connector.male_cutter)

                    if materialized_connector.additional_parts is not None:
                        for additional_part in materialized_connector.additional_parts:
                            connectors = connectors.fuse(additional_part)

                    # local_coord_system = create_coordinate_system(hint.edge_centroid, hint.edge_vector, hint.triangle_a_normal, 10)
                    # collector = collector.fuse(local_coord_system)

                    if (
                        materialized_connector.non_production_parts is not None
                        and False
                    ):  # FIXME
                        if not PROD:
                            for (
                                additional_part
                            ) in materialized_connector.non_production_parts:
                                collector = collector.fuse(additional_part)

                elif region_id == materialized_connector.female_region:

                    if materialized_connector.female_connector is not None:
                        connectors = connectors.fuse(
                            materialized_connector.female_connector
                        )
                    if materialized_connector.female_cutter is not None:
                        cutters = cutters.fuse(materialized_connector.female_cutter)

    if isinstance(connectors, PartCollector):
        connectors = None

    if isinstance(cutters, PartCollector):
        cutters = None
    return connectors, cutters


def coil_origin_and_direction(angle_rad, origin_radius=0, coil_start_angle_offset=0):

    x = origin_radius * math.sin(angle_rad + coil_start_angle_offset)
    y = angle_rad / (2 * math.pi) * coil_pitch
    z = origin_radius * math.cos(angle_rad + coil_start_angle_offset)

    coil_direction = (
        math.cos(angle_rad + coil_start_angle_offset),
        0,
        math.sin(angle_rad + coil_start_angle_offset),
    )

    return (x, y, z), coil_direction


def create_led_strip_coil(partition, local_led_band_head_gap):
    cross_section = np.array(
        [
            [-led_band_width / 2, 0.0],  # Bottom left
            [led_band_width / 2, 0.0],  # Bottom right
            [led_band_width / 2, led_band_thickness],  # Top right
            [-led_band_width / 2, led_band_thickness],  # Top left
        ]
    )

    unit_length_rays_coil = []

    for k in range(coil_turns * resolution):
        angle = 2 * math.pi * k / resolution
        origin, direction = coil_origin_and_direction(
            angle, coil_start_angle_offset=np.radians(coil_start_angle_offset_degrees)
        )

        unit_length_rays_coil.append((origin, origin + normalize(direction)))

    t = coordinate_system_transform_to_matrix(
        {
            "rotation_angle": np.radians(coil_angle),
            "rotation_axis": np.array((1, 0, 0), dtype=np.float64),
            "translation": np.array((0, coil_y_offset, 0), dtype=np.float64),
        }
    )

    unit_length_rays_coil = [
        ((t @ np.array(list(origin) + [1]))[:3], (t @ np.array(list(end) + [1]))[:3])
        for origin, end in unit_length_rays_coil
    ]

    markers = PartCollector()
    coil_points = []
    coil_normals = []

    def current_coil_length(current_coil_points):
        length = 0.0
        for i in range(1, len(current_coil_points)):
            length += np.linalg.norm(
                np.array(current_coil_points[i]) - np.array(current_coil_points[i - 1])
            )
        return length

    complete_partition = MeshPartition(partition.mesh)
    band_relevant_region_view = TransformedRegionView(complete_partition, 0)

    for origin, end in unit_length_rays_coil:

        direction = np.array(end) - np.array(origin)

        hits = band_relevant_region_view.ray_intersect_faces(origin, direction)

        for face_id, intersect in hits:

            if intersect is not None:

                normal = band_relevant_region_view.face_normal(face_id)

                marker_cylinder = directed_cylinder_at(
                    intersect,
                    normal,
                    marker_sphere_radius,
                    10 * marker_sphere_radius,
                )
                markers = markers.fuse(marker_cylinder)

                coil_point = intersect + normal * local_led_band_head_gap
                coil_points.append(coil_point)
                coil_normals.append(normal)

                break
        if current_coil_length(coil_points) > led_band_length:
            break

    meshes = create_trapezoidal_snake_geometry(
        cross_section, coil_points, coil_normals, close_loop=False
    )

    coil_collector = PartCollector()
    for mesh in meshes:
        try:
            solid = create_solid_from_traditional_face_vertex_maps(mesh)
            coil_collector = coil_collector.fuse(solid)
        except Exception as e:
            print(f"Error creating solid from coil segment: {e}")

    return coil_collector, markers


def create_face_grid():

    partition = create_partition()

    region_views = [
        TransformedRegionView(partition, region_id)
        for region_id in partition.get_regions()
    ]

    parts = {}

    eye_regions = partition.find_regions_of_vertex_by_label("eye")
    print(f"Eye regions: {eye_regions}")
    region_ids_to_print = calc_region_ids_to_print(partition, eye_regions)
    print(f"Region ids to print: {region_ids_to_print}")

    connector_region_id_pairs = calc_connector_region_id_pairs(
        partition, region_views, eye_regions
    )

    print(f"Connector region pairs: {connector_region_id_pairs}")

    for region_view in region_views:

        if region_view.region_id not in region_ids_to_print:
            continue

        if LAY_FLAT_FOR_PROD:
            region_view = lay_flat_for_prod(region_view, region_views)

        print(
            f"Region {region_view.region_id} has {region_view.printability_score()} printability score."
        )

        region_id = region_view.region_id

        connector_hints = region_view.compute_transformed_connector_hints(
            shell_thickness,
            merge_connectors=True,
            min_connector_distance=5 * connector_length,
            min_corner_distance=4 * connector_length,
        )

        shell_maps, _ = region_view.get_transformed_materialized_shell_maps(
            shell_thickness=shell_thickness,
            shrinkage=0,
            shrink_border=shrink_border,
            smooth_inside=True,
        )

        collector = PartCollector()
        for maps in shell_maps.values():
            try:
                solid = create_solid_from_traditional_face_vertex_maps(maps)
                collector = collector.fuse(solid)
            except Exception as e:
                print(f"Error creating solid from maps: {e}")

        V, F, E = region_view.get_transformed_vertices_faces_boundary_edges()

        outer_point_vertex_ids = region_view.find_local_vertex_ids_by_label(
            "outer_point_2"
        )

        if outer_point_vertex_ids:

            outer_point_vertex_id = (
                outer_point_vertex_ids[0] if outer_point_vertex_ids else None
            )
            outer_point_vertex = V[outer_point_vertex_ids[0]]

            outer_point_average_normal = region_view.average_normal_at_vertex(
                outer_point_vertex_id
            )

            print(f"Outer point avertage normal: {outer_point_average_normal}")
            transformation_function = coordinate_system_transformation_function(
                (0, 0, 0),
                (0, 0, 1),
                (1, 0, 0),
                outer_point_vertex,
                (1, 0, 0),
                outer_point_average_normal,
                rotate,
                translate,
            )

        if DRILL_AIR_HOLES:
            air_hole_drills = create_air_hole_drills(region_view)
            collector = collector.cut(air_hole_drills)

        if CREATE_EMBELLISHMENTS:

            feature_collector = create_embellishments(region_view)

        if not PROD and SHOW_REGION_IDS:

            collector = collector.fuse(create_region_id_marker(region_view))

        if LAY_FLAT_FOR_PROD:

            z_min = np.min(
                region_view.get_transformed_vertices_faces_boundary_edges()[0][:, 2]
            )

            z_min_of_collector = get_z_min(collector)

            if (
                z_min_of_collector < z_min
                and z_min - z_min_of_collector > connector_length / 2
            ):
                z_min = z_min_of_collector

            collector = translate(0, 0, -z_min)(collector)

            if CREATE_EMBELLISHMENTS:
                feature_collector = translate(0, 0, -z_min)(feature_collector)

                feature_bottom_shaver = create_basic_box(
                    BIG_THING,
                    BIG_THING,
                    BIG_THING,
                    origin=(
                        -BIG_THING / 2,
                        -BIG_THING / 2,
                        -BIG_THING + embellishment_bottom_shaver_thickness,
                    ),
                )

                feature_collector = feature_collector.cut(feature_bottom_shaver)

                collector = collector.fuse(feature_collector)

            collector = translate(0, 0, -bottom_shaver_thickness)(collector)

            bed_cutter = create_basic_box(
                BIG_THING,
                BIG_THING,
                BIG_THING,
                origin=(-BIG_THING / 2, -BIG_THING / 2, -BIG_THING),
            )

            collector = collector.cut(bed_cutter)
        else:
            if CREATE_EMBELLISHMENTS:
                print(f"Fusing feature collector into main collector")
                collector = collector.fuse(feature_collector)

        connectors, cutters = create_connectors_and_connector_cutters(
            connector_hints, region_view, connector_region_id_pairs
        )

        if connectors is not None:
            collector = collector.fuse(connectors)
        if cutters is not None:
            collector = collector.cut(cutters)
        parts[region_id] = collector

    coil_collector, markers = create_led_strip_coil(partition, led_band_head_gap)

    coil_cutter, _ = create_led_strip_coil(partition, -0.2)

    new_parts = {}
    for region_id, part in parts.items():
        part = part.cut(coil_cutter)
        new_parts[region_id] = part
    parts = new_parts

    print(f"Returning {parts}")

    return parts, coil_collector, markers


def create_led_band_holder(cutter_clearance) -> LeaderFollowersCuttersPart:

    the_objects = {}
    for clearance, key in [(0, "holder"), (cutter_clearance, "holder_cutter")]:
        print(f"Creating holder with cutter clearance: {clearance}")
        right_points = np.array(
            [
                (led_band_width / 2 + holder_thickness + clearance, -clearance),
                (
                    led_band_width / 2 + holder_thickness + clearance,
                    led_band_thickness
                    + 2 * holder_vertical_gap
                    + 2 * holder_thickness
                    + clearance,
                ),
                (
                    led_band_width / 2 - holder_clamp_size - clearance,
                    led_band_thickness
                    + 2 * holder_vertical_gap
                    + 2 * holder_thickness
                    + clearance,
                ),
                (
                    led_band_width / 2 - holder_clamp_size - clearance,
                    led_band_thickness
                    + holder_thickness
                    + 2 * holder_vertical_gap
                    + clearance,
                ),
                (
                    led_band_width / 2 + holder_horizontal_gap + clearance,
                    led_band_thickness
                    + holder_thickness
                    + 2 * holder_vertical_gap
                    + clearance,
                ),
                (
                    led_band_width / 2 + holder_horizontal_gap + clearance,
                    holder_thickness + clearance,
                ),
            ]
        )

        left_points = (np.array([[-1, 0], [0, 1]]) @ right_points.T).T

        points = right_points.tolist() + list(reversed(left_points.tolist()))

        print(f"Points: {points}")

        holder = create_extruded_polygon(
            points,
            holder_length + 2 * clearance,
        )
        the_objects[key] = holder

    holder = the_objects["holder"]

    holder_cutter = the_objects["holder_cutter"]
    holder_cutter = align(holder_cutter, holder, Alignment.CENTER)

    screw = create_cylinder_screw(
        holder_screw_size,
        holder_screw_length,
    )

    screw = rotate(90, axis=(1, 0, 0))(screw)
    screw = align(screw, holder, Alignment.CENTER)

    screw = align(screw, holder, Alignment.STACK_FRONT)
    screw = translate(0, holder_thickness + holder_vertical_gap, 0)(screw)

    screw_hole_drill = Part.makeCylinder(
        m_screws_table[holder_screw_size]["core_hole"] / 2, BIG_THING
    )
    screw_hole_drill = rotate(90, axis=(1, 0, 0))(screw_hole_drill)
    screw_hole_drill = align(screw_hole_drill, screw, Alignment.CENTER)
    holder = holder.cut(screw_hole_drill)

    screw_cutter = create_cylinder_screw(
        holder_screw_size,
        holder_screw_length,
        enlargement=cutter_clearance,
    )
    screw_cutter = rotate(90, axis=(1, 0, 0))(screw_cutter)
    screw_cutter = align(screw_cutter, screw, Alignment.CENTER)

    holder = LeaderFollowersCuttersPart(
        holder,
        followers=[],
        cutters=[holder_cutter, screw_cutter],
        non_production_parts=[screw],
    )

    holder = translate(0, -holder_thickness - holder_vertical_gap, 0)(holder)

    return holder


# INIT
print("Initializing FreeCAD...")
init_freecad()
parts = PartList()


try:

    sphere_grid_parts, coil, markers = create_face_grid()

    all_parts = PartCollector()
    first_part = None
    for i, (key, part) in enumerate(sphere_grid_parts.items()):

        print(f"Adding part {key} to parts list, part = {part}")

        center = get_bounding_box_center_np(part)

        EXPLOSION_OFFSET = 0.0  # 0.05
        part = translate(*(center * EXPLOSION_OFFSET))(part)

        all_parts = all_parts.fuse(part)

    # all_parts = all_parts.fuse(coil)
    # all_parts = all_parts.fuse(markers)

    min_z = get_z_min(all_parts)

    all_parts = translate(0, 0, -min_z)(all_parts)

    part_vertices = get_vertex_coordinates(all_parts)
    part_vertices = np.array(part_vertices)

    part_2d_vertices = part_vertices[:, :2]
    part_2d_vertices -= np.mean(part_2d_vertices, axis=0)

    if PROD:
        best_angle = None
        best_score = -np.inf
        for i in range(180):

            angle = i * 2 * np.pi / 180

            rotation_matrix = np.array(
                [
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ]
            )
            rotated_2d_vertices = part_2d_vertices @ rotation_matrix.T

            width = np.max(rotated_2d_vertices[:, 0]) - np.min(
                rotated_2d_vertices[:, 0]
            )
            depth = np.max(rotated_2d_vertices[:, 1]) - np.min(
                rotated_2d_vertices[:, 1]
            )
            score = -np.max([width, depth])

            if score > best_score:
                best_score = score
                best_angle = angle

        print(f"Best angle for rotation: {np.degrees(best_angle)} degrees")

        all_parts = rotate(np.degrees(best_angle))(all_parts)

        all_parts = rotate(90)(all_parts)

    parts.add(
        all_parts,
        f"all_parts",
        skip_in_production=False,
        flip=False,
    )

    arrange_and_export_parts(
        parts.as_list(),
        PROD_GAP,
        BED_WIDTH,
        __file__,
        prod=PROD,
        process_data=PROCESS_DATA,
        max_build_height=MAX_BUILD_HEIGHT,
    )

    elapsed_time = time.time() - overall_start_time
    print(f"Overall script took {elapsed_time:.1f} seconds.")

except Exception as e:
    print(e)
    import traceback

    print(traceback.format_exc())
    raise e
