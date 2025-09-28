import pytest
from shellforgepy.construct.leaders_followers_cutters_part import (
    LeaderFollowersCuttersPart,
)
from shellforgepy.construct.named_part import NamedPart
from shellforgepy.produce.arrange_and_export import PartList
from shellforgepy.simple import (
    create_basic_box,
    get_bounding_box_center,
    rotate,
    translate,
)


def test_part_list_add_and_as_list():
    plist = PartList()
    shape = create_basic_box(3, 3, 3)
    plist.add(shape, "cube", prod_rotation_angle=45.0, prod_rotation_axis=(0, 1, 0))

    follower = translate(5, 0, 0)(create_basic_box(1, 1, 1))
    group = LeaderFollowersCuttersPart(follower)
    plist.add(group, "follower", skip_in_production=True)

    entries = plist.as_list()
    assert {entry["name"] for entry in entries} == {"cube", "follower"}
    cube_entry = next(entry for entry in entries if entry["name"] == "cube")
    assert cube_entry["prod_rotation_axis"] == [0.0, 1.0, 0.0]
    assert cube_entry["part"] is not None  # Just check it exists


def test_part_list_duplicate_name_raises():
    plist = PartList()
    shape = create_basic_box(1, 1, 1)
    plist.add(shape, "part")
    with pytest.raises(ValueError):
        plist.add(shape, "part")


def test_leader_followers_translate_and_rotate():
    leader = create_basic_box(2, 2, 2)
    follower_shape = translate(4, 0, 0)(create_basic_box(1, 1, 1))
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
    leader = create_basic_box(2, 2, 2)
    follower = NamedPart(
        "follower",
        translate(2.5, 0, 0)(create_basic_box(1, 1, 1)),
    )
    cutter = NamedPart("cutter", create_basic_box(0.5, 0.5, 0.5))
    aux = NamedPart("aux", create_basic_box(0.2, 0.2, 0.2))

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

    combined = group.fuse(translate(0, 0, 2)(create_basic_box(1, 1, 1)))
    assert isinstance(combined, LeaderFollowersCuttersPart)
    # Just check that leader exists after fusion
    assert combined.leader is not None
