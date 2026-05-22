from pathlib import Path

import pytest
from shellforgepy.metrics import (
    BuildingKind,
    Material,
    build_building_cost_report_lines,
    build_building_cost_totals_by_assembly_id,
    build_ground_coverage_report_lines,
    build_living_space_report_lines,
    build_living_space_totals_by_assembly_id,
    build_metrics_report_lines,
    configure_building_cost_per_m3_map,
    merge_metrics_snapshot,
    record_building_cost_metric,
    record_ground_coverage_footprint_metric,
    record_ground_coverage_phase_metric,
    record_length_metric,
    record_living_space_metric,
    record_mark_metric,
    record_measured_mass_metric,
    record_weight_metric,
    reset_metrics,
    snapshot_has_metrics,
    snapshot_metrics,
    write_metrics_report,
)


def setup_function():
    reset_metrics()


def teardown_function():
    reset_metrics()


def test_build_metrics_report_lines_groups_duplicate_lengths():
    record_length_metric(
        "extrusion_profile",
        "4040",
        "left_z_axis_profile",
        550,
    )
    record_length_metric(
        "extrusion_profile",
        "4040",
        "right_z_axis_profile",
        550.0,
    )
    record_length_metric(
        "linear_rail",
        "MGN12",
        "x_axis_rail",
        450,
    )

    assert build_metrics_report_lines() == [
        "Cut stock metrics:",
        "extrusion_profile 4040:",
        "  550 mm x2",
        "    - left_z_axis_profile",
        "    - right_z_axis_profile",
        "linear_rail MGN12:",
        "  450 mm x1",
        "    - x_axis_rail",
    ]


def test_build_metrics_report_lines_includes_weight_breakdown():
    record_weight_metric(
        "y_axis_moving_mass",
        Material.ALUMINUM,
        1000,
        part_id="print_bed_main",
    )
    record_weight_metric(
        "y_axis_moving_mass",
        Material.STEEL,
        1000,
        part_id="print_bed_mount_screws",
    )
    record_weight_metric(
        "y_axis_moving_mass",
        Material.STEEL,
        1000,
        part_id="print_bed_mount_screws",
    )

    assert build_metrics_report_lines() == [
        "Weight metrics:",
        "y_axis_moving_mass: 0.018400 kg",
        "  ALUMINUM: 0.002700 kg",
        "  STEEL: 0.015700 kg",
        "  print_bed_main (ALUMINUM): 0.002700 kg",
        "  print_bed_mount_screws (STEEL): 0.015700 kg",
    ]


def test_building_cost_metrics_are_grouped_by_assembly_kind_and_part():
    configure_building_cost_per_m3_map(
        {
            "garage": 600,
            BuildingKind.APPARTMENT: 1100,
        }
    )
    record_building_cost_metric(
        "extension_building_cost",
        BuildingKind.APPARTMENT,
        10.0,
        part_id="storey_1",
    )
    record_building_cost_metric(
        "extension_building_cost",
        "appartment",
        5.0,
        part_id="storey_2",
    )
    record_building_cost_metric(
        "fan_garage_building_cost",
        "garage",
        2.0,
        part_id="fan_garage",
    )

    assert build_building_cost_totals_by_assembly_id() == {
        "extension_building_cost": 16500.0,
        "fan_garage_building_cost": 1200.0,
    }
    assert build_building_cost_report_lines() == [
        "Building cost metrics:",
        "Overview: 17.000 m3, 18k",
        "extension_building_cost: 15.000 m3, 16k",
        "  APPARTMENT: 15.000 m3, 16k",
        "  storey_1 (APPARTMENT): 10.000 m3 @ 1100/m3 = 11k",
        "  storey_2 (APPARTMENT): 5.000 m3 @ 1100/m3 = 6k",
        "fan_garage_building_cost: 2.000 m3, 1k",
        "  GARAGE: 2.000 m3, 1k",
        "  fan_garage (GARAGE): 2.000 m3 @ 600/m3 = 1k",
    ]


def test_building_cost_metric_stores_resolved_rate_when_recorded():
    configure_building_cost_per_m3_map({"appartment": 100})
    record_building_cost_metric(
        "extension_building_cost",
        BuildingKind.APPARTMENT,
        2.0,
        part_id="storey_1",
    )
    configure_building_cost_per_m3_map({"appartment": 999})

    assert build_building_cost_report_lines() == [
        "Building cost metrics:",
        "Overview: 2.000 m3, 0k",
        "extension_building_cost: 2.000 m3, 0k",
        "  APPARTMENT: 2.000 m3, 0k",
        "  storey_1 (APPARTMENT): 2.000 m3 @ 100/m3 = 0k",
    ]


def test_living_space_metrics_are_grouped_by_assembly_and_part():
    record_living_space_metric(
        "extension_living_space",
        10.0,
        part_id="storey_1",
    )
    record_living_space_metric(
        "extension_living_space",
        5.5,
        part_id="storey_2",
    )

    assert build_living_space_totals_by_assembly_id() == {
        "extension_living_space": 15.5
    }
    assert build_living_space_report_lines() == [
        "Living space metrics:",
        "Overview: 15.500 m2",
        "extension_living_space: 15.500 m2",
        "  storey_1: 10.000 m2",
        "  storey_2: 5.500 m2",
    ]


def test_ground_coverage_metrics_report_phase_ratios_and_footprints():
    record_ground_coverage_footprint_metric("before", "existing_house", 100.0)
    record_ground_coverage_footprint_metric("before", "existing_garage", 25.0)
    record_ground_coverage_footprint_metric("after", "existing_house", 100.0)
    record_ground_coverage_footprint_metric("after", "extension", 40.0)
    record_ground_coverage_phase_metric(
        "before",
        property_square_meters=500.0,
        covered_square_meters=120.0,
    )
    record_ground_coverage_phase_metric(
        "after",
        property_square_meters=500.0,
        covered_square_meters=135.0,
    )

    assert build_ground_coverage_report_lines() == [
        "Ground coverage metrics:",
        "Property area: 500.000 m2",
        "before: 120.000 m2 / 500.000 m2 = 24.00%",
        "  existing_house: 100.000 m2",
        "  existing_garage: 25.000 m2",
        "after: 135.000 m2 / 500.000 m2 = 27.00%",
        "  existing_house: 100.000 m2",
        "  extension: 40.000 m2",
    ]


def test_ground_coverage_metrics_report_legal_limit_remaining_and_overage():
    record_ground_coverage_footprint_metric("before", "existing_house", 100.0)
    record_ground_coverage_footprint_metric("after", "existing_house", 100.0)
    record_ground_coverage_phase_metric(
        "before",
        property_square_meters=500.0,
        covered_square_meters=120.0,
        max_coverage_ratio=0.25,
    )
    record_ground_coverage_phase_metric(
        "after",
        property_square_meters=500.0,
        covered_square_meters=135.0,
        max_coverage_ratio=0.25,
    )

    assert build_ground_coverage_report_lines() == [
        "Ground coverage metrics:",
        "Property area: 500.000 m2",
        "Legal coverage limit: 25.00% = 125.000 m2",
        "before: 120.000 m2 / 500.000 m2 = 24.00% (5.000 m2 remaining)",
        "  existing_house: 100.000 m2",
        "after: 135.000 m2 / 500.000 m2 = 27.00% (10.000 m2 over limit)",
        "  existing_house: 100.000 m2",
    ]


def test_building_and_living_space_metrics_round_trip_through_snapshots():
    configure_building_cost_per_m3_map({"appartment": 1100})
    record_building_cost_metric(
        "extension_building_cost",
        BuildingKind.APPARTMENT,
        10.0,
        part_id="storey_1",
    )
    record_living_space_metric(
        "extension_living_space",
        42.0,
        part_id="storey_1",
    )

    snapshot = snapshot_metrics()
    reset_metrics()
    merge_metrics_snapshot(snapshot)

    assert build_metrics_report_lines() == [
        "Building cost metrics:",
        "Overview: 10.000 m3, 11k",
        "extension_building_cost: 10.000 m3, 11k",
        "  APPARTMENT: 10.000 m3, 11k",
        "  storey_1 (APPARTMENT): 10.000 m3 @ 1100/m3 = 11k",
        "",
        "Living space metrics:",
        "Overview: 42.000 m2",
        "extension_living_space: 42.000 m2",
        "  storey_1: 42.000 m2",
    ]


def test_ground_coverage_metrics_round_trip_through_snapshots():
    record_ground_coverage_footprint_metric("before", "existing_house", 100.0)
    record_ground_coverage_phase_metric(
        "before",
        property_square_meters=500.0,
        covered_square_meters=100.0,
        max_coverage_ratio=0.25,
    )

    snapshot = snapshot_metrics()
    reset_metrics()
    merge_metrics_snapshot(snapshot)

    assert build_metrics_report_lines() == [
        "Ground coverage metrics:",
        "Property area: 500.000 m2",
        "Legal coverage limit: 25.00% = 125.000 m2",
        "before: 100.000 m2 / 500.000 m2 = 20.00% (25.000 m2 remaining)",
        "  existing_house: 100.000 m2",
    ]


def test_legacy_ground_coverage_snapshot_without_legal_limit_is_supported():
    legacy_snapshot = {
        "schema_version": 3,
        "length_metrics": [],
        "mark_metrics": [],
        "weight_metrics": [],
        "building_cost_per_m3_map": {},
        "building_cost_metrics": [],
        "living_space_metrics": [],
        "ground_coverage_footprint_metrics": [
            {
                "phase": "before",
                "part_id": "existing_house",
                "square_meters": 100.0,
            }
        ],
        "ground_coverage_phase_metrics": [
            {
                "phase": "before",
                "property_square_meters": 500.0,
                "covered_square_meters": 100.0,
            }
        ],
    }

    merge_metrics_snapshot(legacy_snapshot)

    assert build_metrics_report_lines() == [
        "Ground coverage metrics:",
        "Property area: 500.000 m2",
        "before: 100.000 m2 / 500.000 m2 = 20.00%",
        "  existing_house: 100.000 m2",
    ]
    assert snapshot_metrics()["ground_coverage_phase_metrics"] == [
        {
            "phase": "before",
            "property_square_meters": 500.0,
            "covered_square_meters": 100.0,
            "max_coverage_ratio": None,
        }
    ]


def test_legacy_metrics_snapshot_without_building_fields_is_supported():
    legacy_snapshot = {
        "schema_version": 1,
        "length_metrics": [],
        "mark_metrics": [],
        "weight_metrics": [],
    }

    assert snapshot_has_metrics(legacy_snapshot) is False
    merge_metrics_snapshot(legacy_snapshot)
    assert build_metrics_report_lines() == ["Cut stock metrics: no metrics recorded."]


def test_building_cost_metrics_reject_invalid_values():
    with pytest.raises(ValueError, match="not configured"):
        record_building_cost_metric("extension_building_cost", "appartment", 1.0)

    with pytest.raises(ValueError, match="non-negative"):
        configure_building_cost_per_m3_map({"garage": -1})

    configure_building_cost_per_m3_map({"garage": 600})

    with pytest.raises(ValueError, match="assembly_id"):
        record_building_cost_metric("", BuildingKind.GARAGE, 1.0)

    with pytest.raises(ValueError, match="non-negative"):
        record_building_cost_metric("fan_garage_building_cost", "garage", -1.0)

    with pytest.raises(ValueError, match="unsupported building_kind"):
        record_building_cost_metric("fan_garage_building_cost", "castle", 1.0)


def test_metrics_snapshot_can_be_merged_back_into_live_state():
    record_mark_metric(
        stock_type="2020",
        part_name="x_axis_lower_profile",
        stock_length_mm=600,
        mark_name="mount_shield_mount_screw_left",
        position_mm=87.6,
    )
    record_measured_mass_metric(
        "y_axis_moving_mass",
        Material.ALUMINUM,
        0.760,
        part_id="print_bed_main",
    )

    snapshot = snapshot_metrics()
    reset_metrics()

    assert snapshot_has_metrics(snapshot) is True

    merge_metrics_snapshot(snapshot)
    assert build_metrics_report_lines() == [
        "Cut stock metrics:",
        "Stock marks:",
        "x_axis_lower_profile (2020, 600 mm):",
        "  mark at 88 mm - mount_shield_mount_screw_left",
        "",
        "Weight metrics:",
        "y_axis_moving_mass: 0.760000 kg",
        "  ALUMINUM: 0.760000 kg",
        "  print_bed_main (ALUMINUM): 0.760000 kg",
    ]


def test_write_metrics_report_writes_text_artifact(tmp_path):
    record_length_metric("linear_rail", "MGN12", "x_axis_rail", 450)

    report_path = write_metrics_report(tmp_path, base_name="printer_metrics_report")

    assert report_path == Path(tmp_path / "printer_metrics_report.txt").resolve()
    assert report_path.read_text(encoding="utf-8") == "\n".join(
        [
            "Cut stock metrics:",
            "linear_rail MGN12:",
            "  450 mm x1",
            "    - x_axis_rail",
            "",
        ]
    )
