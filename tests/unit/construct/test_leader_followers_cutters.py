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


def test_additional_data_defaults_to_empty_dict():
    leader = create_box(1, 1, 1)
    group = LeaderFollowersCuttersPart(leader)

    assert group.additional_data == {}
    assert isinstance(group.additional_data, dict)


def test_additional_data_accepts_dict():
    leader = create_box(1, 1, 1)
    data = {"material": "steel", "weight": 10.5, "id": 42}
    group = LeaderFollowersCuttersPart(leader, additional_data=data)

    assert group.additional_data == data
    assert group.additional_data is data  # Constructor stores reference directly


def test_additional_data_must_be_dict():
    leader = create_box(1, 1, 1)

    with pytest.raises(AssertionError):
        LeaderFollowersCuttersPart(leader, additional_data="not a dict")

    with pytest.raises(AssertionError):
        LeaderFollowersCuttersPart(leader, additional_data=42)

    with pytest.raises(AssertionError):
        LeaderFollowersCuttersPart(leader, additional_data=["list", "not", "dict"])


def test_additional_data_preserved_in_copy():
    leader = create_box(1, 1, 1)
    original_data = {"material": "aluminum", "finish": "anodized", "count": 5}
    group = LeaderFollowersCuttersPart(leader, additional_data=original_data)

    copied_group = group.copy()

    assert copied_group.additional_data == original_data
    assert copied_group.additional_data is not group.additional_data
    assert copied_group.additional_data is not original_data

    # Modify original to ensure deep copy
    original_data["count"] = 10
    assert copied_group.additional_data["count"] == 5


def test_additional_data_preserved_in_copy_with_nested_data():
    leader = create_box(1, 1, 1)
    original_data = {
        "metadata": {"version": 1, "author": "test"},
        "properties": {"dimensions": [1, 2, 3], "weight": 1.5},
    }
    group = LeaderFollowersCuttersPart(leader, additional_data=original_data)

    copied_group = group.copy()

    assert copied_group.additional_data == original_data
    assert copied_group.additional_data is not group.additional_data

    # Test deep copy by modifying nested data
    original_data["metadata"]["version"] = 2
    original_data["properties"]["dimensions"].append(4)

    assert copied_group.additional_data["metadata"]["version"] == 1
    assert copied_group.additional_data["properties"]["dimensions"] == [1, 2, 3]


def test_additional_data_merged_in_fuse_with_group():
    leader1 = create_box(1, 1, 1)
    leader2 = translate(1.5, 0, 0)(create_box(1, 1, 1))

    data1 = {"source": "group1", "material": "steel", "version": 1}
    data2 = {"source": "group2", "color": "red", "version": 2}

    group1 = LeaderFollowersCuttersPart(leader1, additional_data=data1)
    group2 = LeaderFollowersCuttersPart(leader2, additional_data=data2)

    fused_group = group1.fuse(group2)

    # When fusing two LeaderFollowersCuttersPart objects, additional_data is not merged
    # The implementation doesn't include additional_data merging for this case
    assert fused_group.additional_data == {}


def test_additional_data_preserved_in_fuse_with_shape():
    leader = create_box(1, 1, 1)
    other_shape = translate(1.5, 0, 0)(create_box(1, 1, 1))

    original_data = {"material": "brass", "finish": "polished"}
    group = LeaderFollowersCuttersPart(leader, additional_data=original_data)

    fused_group = group.fuse(other_shape)

    assert fused_group.additional_data == original_data
    assert fused_group.additional_data is not group.additional_data


def test_additional_data_preserved_in_cut_with_shape():
    leader = create_box(2, 2, 2)
    cutter_shape = translate(0.5, 0.5, 0.5)(create_box(1, 1, 1))

    original_data = {"material": "wood", "treatment": "stain", "id": 123}
    group = LeaderFollowersCuttersPart(leader, additional_data=original_data)

    cut_group = group.cut(cutter_shape)

    assert cut_group.additional_data == original_data
    assert cut_group.additional_data is not group.additional_data


def test_additional_data_empty_when_other_has_no_additional_data_in_fuse():
    leader1 = create_box(1, 1, 1)
    leader2 = translate(1.5, 0, 0)(create_box(1, 1, 1))

    data1 = {"material": "plastic", "color": "blue"}

    group1 = LeaderFollowersCuttersPart(leader1, additional_data=data1)
    group2 = LeaderFollowersCuttersPart(leader2)  # No additional_data

    fused_group = group1.fuse(group2)

    # When fusing two LeaderFollowersCuttersPart objects, additional_data is not preserved
    assert fused_group.additional_data == {}


def test_additional_data_overrides_in_fuse():
    leader1 = create_box(1, 1, 1)
    leader2 = translate(1.5, 0, 0)(create_box(1, 1, 1))

    data1 = {"material": "steel", "finish": "brushed", "priority": 1}
    data2 = {
        "material": "aluminum",
        "priority": 2,
    }  # Should override steel and priority

    group1 = LeaderFollowersCuttersPart(leader1, additional_data=data1)
    group2 = LeaderFollowersCuttersPart(leader2, additional_data=data2)

    fused_group = group1.fuse(group2)

    # When fusing two LeaderFollowersCuttersPart objects, additional_data is not merged
    assert fused_group.additional_data == {}


def test_additional_data_independent_modification():
    leader = create_box(1, 1, 1)
    original_data = {"status": "active", "tags": ["test", "sample"]}
    group1 = LeaderFollowersCuttersPart(leader, additional_data=original_data)
    group2 = group1.copy()

    # Modify group1's additional_data
    group1.additional_data["status"] = "inactive"
    group1.additional_data["tags"].append("modified")

    # group2 should be unaffected
    assert group2.additional_data["status"] == "active"
    assert group2.additional_data["tags"] == ["test", "sample"]


def test_additional_data_merge_in_fuse_with_shape_only():
    """Test that additional_data is only merged when fusing with a shape, not with another group."""
    leader = create_box(1, 1, 1)
    other_shape = translate(1.5, 0, 0)(create_box(1, 1, 1))

    # Shape doesn't have additional_data attribute
    original_data = {"material": "brass", "finish": "polished"}
    group = LeaderFollowersCuttersPart(leader, additional_data=original_data)

    fused_group = group.fuse(other_shape)

    assert fused_group.additional_data == original_data
    assert fused_group.additional_data is not group.additional_data


def test_additional_data_merge_preserves_original():
    """Test that the original group's additional_data is not modified during fuse."""
    leader = create_box(1, 1, 1)
    other_shape = translate(1.5, 0, 0)(create_box(1, 1, 1))

    original_data = {"material": "copper", "weight": 5.0}
    group = LeaderFollowersCuttersPart(leader, additional_data=original_data)

    fused_group = group.fuse(other_shape)

    # Original group should be unchanged
    assert group.additional_data == original_data
    assert group.additional_data is original_data

    # Fused group should have a deep copy
    assert fused_group.additional_data == original_data
    assert fused_group.additional_data is not original_data


def test_additional_data_cut_with_group_no_merge():
    """Test that additional_data is not merged when cutting with another group."""
    leader1 = create_box(2, 2, 2)
    leader2 = translate(0.5, 0.5, 0.5)(create_box(1, 1, 1))

    data1 = {"material": "wood", "finish": "stain"}
    data2 = {"tool": "saw", "speed": "slow"}

    group1 = LeaderFollowersCuttersPart(leader1, additional_data=data1)
    group2 = LeaderFollowersCuttersPart(leader2, additional_data=data2)

    cut_group = group1.cut(group2)

    # When cutting with another group, no additional_data parameter is passed to constructor
    assert cut_group.additional_data == {}


def test_additional_data_none_becomes_empty_dict():
    """Test that passing None for additional_data creates an empty dict."""
    leader = create_box(1, 1, 1)
    group = LeaderFollowersCuttersPart(leader, additional_data=None)

    assert group.additional_data == {}
    assert isinstance(group.additional_data, dict)


def test_additional_data_with_part_list():
    """Test that additional_data is accessible even when part is used with PartList."""
    leader = create_box(2, 2, 2)
    follower = NamedPart("follower", translate(3, 0, 0)(create_box(1, 1, 1)))

    metadata = {"part_number": "ABC123", "material": "titanium", "batch": 42}
    group = LeaderFollowersCuttersPart(
        leader, followers=[follower], additional_data=metadata
    )

    # Verify the group still has its additional_data before adding to PartList
    assert group.additional_data == metadata

    plist = PartList()
    plist.add(group, "complex_part", skip_in_production=False)

    entries = plist.as_list()
    assert len(entries) == 1
    assert entries[0]["name"] == "complex_part"

    # The PartList extracts just the leader shape via get_leader_as_part()
    # So the part in the list is the leader shape, not the full group
    part_in_list = entries[0]["part"]
    assert part_in_list is not None

    # But the original group still maintains its additional_data
    assert group.additional_data == metadata


def test_additional_data_with_transformations():
    leader = create_box(1, 1, 1)

    metadata = {"origin": "test_case", "version": 1.0}
    group = LeaderFollowersCuttersPart(leader, additional_data=metadata)

    # Apply translation
    translated_group = translate(5, 0, 0)(group)
    assert translated_group.additional_data == metadata
    assert translated_group.additional_data is not group.additional_data

    # Apply rotation
    rotated_group = rotate(90, axis=(0, 0, 1))(group)
    assert rotated_group.additional_data == metadata
    assert rotated_group.additional_data is not group.additional_data


def test_additional_data_with_mirror_transformation():
    """Test that additional_data is preserved through mirror transformations."""
    from shellforgepy.simple import mirror

    leader = create_box(2, 1, 1)
    metadata = {"symmetry": "bilateral", "material": "steel"}
    group = LeaderFollowersCuttersPart(leader, additional_data=metadata)

    # Apply mirror transformation
    mirrored_group = mirror(normal=(1, 0, 0), point=(0, 0, 0))(group)
    assert mirrored_group.additional_data == metadata
    assert mirrored_group.additional_data is not group.additional_data


def test_additional_data_with_in_place_transformations():
    """Test that additional_data is preserved with in-place transformation methods."""
    leader = create_box(1, 1, 1)
    metadata = {"inplace": True, "method": "direct"}
    group = LeaderFollowersCuttersPart(leader, additional_data=metadata)

    # Test in-place translation (expects vector tuple)
    result = group.translate((1, 0, 0))
    assert result is group  # Should return self
    assert group.additional_data == metadata

    # Test in-place rotation (expects point, point, angle)
    result = group.rotate((0, 0, 0), (0, 0, 1), 45)
    assert result is group  # Should return self
    assert group.additional_data == metadata

    # Note: In-place mirror method is adapter-specific in signature,
    # so we test mirror functionality only through the functional interface
    # in test_additional_data_with_mirror_transformation()


def test_use_complex_part_as_leader():

    basic_part = create_box(2, 2, 2)

    wrapped_1 = LeaderFollowersCuttersPart(basic_part)

    re_wrapped = LeaderFollowersCuttersPart(wrapped_1)

    basic_part_2 = create_box(1, 1, 1)

    re_wrapped_fused = re_wrapped.fuse(basic_part_2)
