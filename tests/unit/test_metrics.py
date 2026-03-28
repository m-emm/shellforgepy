from pathlib import Path

from shellforgepy.metrics import (
    Material,
    build_metrics_report_lines,
    merge_metrics_snapshot,
    record_length_metric,
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
