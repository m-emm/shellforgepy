import pytest
from shellforgepy.construct.leader_followers_cutters_part import (
    LeaderFollowersCuttersPart,
)
from shellforgepy.construct.named_part import NamedPart
from shellforgepy.produce.arrange_and_export import PartList
from shellforgepy.simple import (
    create_box,
    get_bounding_box,
    get_bounding_box_center,
    get_volume,
    rotate,
    translate,
)


def test_part_list_add_and_as_list():
    plist = PartList()
    shape = create_box(3, 3, 3)
    plist.add(shape, "cube", prod_rotation_angle=45.0, prod_rotation_axis=(0, 1, 0))

    follower = translate(5, 0, 0)(create_box(1, 1, 1))
    group = LeaderFollowersCuttersPart(follower)
    plist.add(group, "follower", skip_in_production=True)

    entries = plist.as_list()
    assert {entry["name"] for entry in entries} == {"cube", "follower"}
    cube_entry = next(entry for entry in entries if entry["name"] == "cube")
    assert cube_entry["prod_rotation_axis"] == [0.0, 1.0, 0.0]
    assert cube_entry["part"] is not None  # Just check it exists


def test_part_list_duplicate_name_raises():
    plist = PartList()
    shape = create_box(1, 1, 1)
    plist.add(shape, "part")
    with pytest.raises(ValueError):
        plist.add(shape, "part")


def test_leader_followers_translate_and_rotate():
    leader = create_box(2, 2, 2)
    follower_shape = translate(4, 0, 0)(create_box(1, 1, 1))
    follower = NamedPart("follower", follower_shape)
    group = LeaderFollowersCuttersPart(leader, followers=[follower])

    original_leader_center = get_bounding_box_center(group.leader)
    original_follower_center = get_bounding_box_center(group.followers[0].part)

    group.translate((5, 0, 0))

    translated_leader_center = get_bounding_box_center(group.leader)
    translated_follower_center = get_bounding_box_center(group.followers[0].part)

    assert translated_leader_center[0] == pytest.approx(original_leader_center[0] + 5)
    assert translated_follower_center[0] == pytest.approx(
        original_follower_center[0] + 5
    )

    # Use functional interface for framework-standardized parameters
    group = rotate(90, center=(0, 0, 0), axis=(0, 0, 1))(group)
    rotated_leader_center = get_bounding_box_center(group.leader)

    # A 90° rotation around Z should transform (x, y, z) -> (-y, x, z)
    # So if translated_center is (6, 1, 1), rotated should be (-1, 6, 1)
    expected_x = -translated_leader_center[1]  # -1
    expected_y = translated_leader_center[0]  # 6

    assert rotated_leader_center[0] == pytest.approx(expected_x, abs=1e-6)
    assert rotated_leader_center[1] == pytest.approx(expected_y, abs=1e-6)


def test_leader_followers_fuse_and_non_production():
    leader = create_box(2, 2, 2)
    follower = NamedPart(
        "follower",
        translate(2.5, 0, 0)(create_box(1, 1, 1)),
    )
    cutter = NamedPart("cutter", create_box(0.5, 0.5, 0.5))
    aux = NamedPart("aux", create_box(0.2, 0.2, 0.2))

    group = LeaderFollowersCuttersPart(
        leader,
        followers=[follower],
        cutters=[cutter],
        non_production_parts=[aux],
    )

    fused = group.leaders_followers_fused()
    # For now, just check that fusion works and returns something
    assert fused is not None

    non_prod_fused = group.get_non_production_parts_fused()
    assert non_prod_fused is not None

    combined = group.fuse(translate(0, 0, 2)(create_box(1, 1, 1)))
    assert isinstance(combined, LeaderFollowersCuttersPart)
    # Just check that leader exists after fusion
    assert combined.leader is not None


def test_leader_followers_cut_with_group_merges_metadata():
    leader = create_box(2, 2, 2)
    follower_a = NamedPart(
        "follower_a",
        translate(3, 0, 0)(create_box(1, 1, 1)),
    )
    cutter_a = NamedPart("cutter_a", create_box(0.5, 0.5, 0.5))
    aux_a = NamedPart("aux_a", create_box(0.25, 0.25, 0.25))
    group_a = LeaderFollowersCuttersPart(
        leader,
        followers=[follower_a],
        cutters=[cutter_a],
        non_production_parts=[aux_a],
    )

    subtractor_leader = translate(0.5, 0.5, 0.5)(create_box(1, 1, 1))
    follower_b = NamedPart(
        "follower_b",
        translate(-3, 0, 0)(create_box(1, 1, 1)),
    )
    cutter_b = NamedPart("cutter_b", create_box(0.25, 0.25, 0.25))
    aux_b = NamedPart("aux_b", create_box(0.2, 0.2, 0.2))
    group_b = LeaderFollowersCuttersPart(
        subtractor_leader,
        followers=[follower_b],
        cutters=[cutter_b],
        non_production_parts=[aux_b],
    )

    result = group_a.cut(group_b)

    assert isinstance(result, LeaderFollowersCuttersPart)
    assert len(result.followers) == 2
    assert len(result.cutters) == 2
    assert len(result.non_production_parts) == 2

    original_volume = get_volume(leader)
    subtractor_volume = get_volume(subtractor_leader)
    assert get_volume(result.leader) == pytest.approx(
        original_volume - subtractor_volume, rel=1e-6
    )

    follower_a_clone = next(
        part for part in result.followers if part.name == "follower_a"
    )
    follower_b_clone = next(
        part for part in result.followers if part.name == "follower_b"
    )
    cutter_a_clone = next(part for part in result.cutters if part.name == "cutter_a")
    cutter_b_clone = next(part for part in result.cutters if part.name == "cutter_b")
    aux_a_clone = next(
        part for part in result.non_production_parts if part.name == "aux_a"
    )
    aux_b_clone = next(
        part for part in result.non_production_parts if part.name == "aux_b"
    )

    assert follower_a_clone is not group_a.followers[0]
    assert follower_b_clone is not group_b.followers[0]
    assert cutter_a_clone is not group_a.cutters[0]
    assert cutter_b_clone is not group_b.cutters[0]
    assert aux_a_clone is not group_a.non_production_parts[0]
    assert aux_b_clone is not group_b.non_production_parts[0]

    assert follower_a_clone.part is not group_a.followers[0].part
    assert follower_b_clone.part is not group_b.followers[0].part


def test_leader_followers_cut_with_shape_preserves_metadata():
    leader = create_box(2, 2, 2)
    follower = NamedPart(
        "follower",
        translate(3, 0, 0)(create_box(1, 1, 1)),
    )
    cutter = NamedPart("cutter", create_box(0.5, 0.5, 0.5))
    aux = NamedPart("aux", create_box(0.25, 0.25, 0.25))
    group = LeaderFollowersCuttersPart(
        leader,
        followers=[follower],
        cutters=[cutter],
        non_production_parts=[aux],
    )

    subtractor = translate(0.5, 0.5, 0.5)(create_box(1, 1, 1))
    result = group.cut(subtractor)

    assert isinstance(result, LeaderFollowersCuttersPart)
    assert len(result.followers) == 1
    assert len(result.cutters) == 1
    assert len(result.non_production_parts) == 1

    original_volume = get_volume(leader)
    subtractor_volume = get_volume(subtractor)
    assert get_volume(result.leader) == pytest.approx(
        original_volume - subtractor_volume, rel=1e-6
    )

    follower_clone = result.followers[0]
    cutter_clone = result.cutters[0]
    aux_clone = result.non_production_parts[0]

    assert follower_clone is not group.followers[0]
    assert cutter_clone is not group.cutters[0]
    assert aux_clone is not group.non_production_parts[0]

    assert follower_clone.part is not group.followers[0].part


def test_leader_followers_cut_requires_cuttable_other():
    class DummyPart:
        pass

    group = LeaderFollowersCuttersPart(DummyPart())

    with pytest.raises(TypeError):
        group.cut(object())


def test_leader_followers_boundbox_property_matches_leader_bounds():
    leader = translate(1, -2, 3)(create_box(2, 4, 6))
    group = LeaderFollowersCuttersPart(leader)

    expected_min, expected_max = get_bounding_box(leader)

    bound_box = group.BoundBox

    assert not callable(bound_box)
    assert bound_box.XMin == pytest.approx(expected_min[0])
    assert bound_box.YMin == pytest.approx(expected_min[1])
    assert bound_box.ZMin == pytest.approx(expected_min[2])
    assert bound_box.XMax == pytest.approx(expected_max[0])
    assert bound_box.YMax == pytest.approx(expected_max[1])
    assert bound_box.ZMax == pytest.approx(expected_max[2])
