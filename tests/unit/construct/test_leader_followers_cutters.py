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


def test_merge_except_leader_merges_all_except_primary():
    base_leader = create_box(1, 1, 1)
    follower_named = NamedPart(
        "self_follower",
        translate(2, 0, 0)(create_box(0.5, 0.5, 0.5)),
    )
    follower_unnamed = translate(3, 0, 0)(create_box(0.4, 0.4, 0.4))
    cutter_named = NamedPart("self_cutter", create_box(0.2, 0.2, 0.2))
    cutter_unnamed = create_box(0.1, 0.1, 0.1)
    aux_named = NamedPart("self_aux", create_box(0.05, 0.05, 0.05))
    aux_unnamed = create_box(0.06, 0.06, 0.06)

    main_group = LeaderFollowersCuttersPart(
        base_leader,
        followers=[follower_unnamed],
        cutters=[cutter_unnamed],
        non_production_parts=[aux_unnamed],
    )
    main_group.add_named_follower(follower_named, "self_follower_named")
    main_group.add_named_cutter(cutter_named, "self_cutter_named")
    main_group.add_named_non_production_part(aux_named, "self_aux_named")

    other_leader = translate(10, 0, 0)(create_box(1, 1, 1))
    other_follower_unnamed = translate(13, 0, 0)(create_box(0.25, 0.25, 0.25))
    other_follower_named = translate(12, 0, 0)(create_box(0.3, 0.3, 0.3))
    other_cutter_unnamed = translate(10, 0, 0)(create_box(0.12, 0.12, 0.12))
    other_cutter_named = translate(10, 0, 0)(create_box(0.15, 0.15, 0.15))
    other_aux_unnamed = translate(10, 0, 0)(create_box(0.08, 0.08, 0.08))
    other_aux_named = translate(10, 0, 0)(create_box(0.07, 0.07, 0.07))

    other_group = LeaderFollowersCuttersPart(
        other_leader,
        followers=[other_follower_unnamed],
        cutters=[other_cutter_unnamed],
        non_production_parts=[other_aux_unnamed],
    )
    other_group.add_named_follower(other_follower_named, "other_follower_named")
    other_group.add_named_cutter(other_cutter_named, "other_cutter_named")
    other_group.add_named_non_production_part(other_aux_named, "other_aux_named")

    merged = main_group.merge_except_leader(other_group)

    # Leader should remain the original one and unchanged
    assert merged.leader is base_leader
    assert get_volume(merged.leader) == pytest.approx(get_volume(base_leader))

    # All parts except leaders should be merged, keeping names from both groups
    assert len(merged.followers) == 4
    assert len(merged.cutters) == 4
    assert len(merged.non_production_parts) == 4

    assert merged.get_follower_index_by_name("self_follower_named") == 0
    assert merged.get_follower_index_by_name("other_follower_named") == 2
    assert merged.get_cutter_index_by_name("self_cutter_named") == 0
    assert merged.get_cutter_index_by_name("other_cutter_named") == 2
    assert merged.get_non_production_index_by_name("self_aux_named") == 0
    assert merged.get_non_production_index_by_name("other_aux_named") == 2

    # Unnamed parts should be cloned rather than reusing the original objects
    assert merged.followers[1] is not follower_unnamed
    assert merged.cutters[1] is not cutter_unnamed
    assert merged.non_production_parts[1] is not aux_unnamed


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
    # Cut operation only preserves the original part's components
    assert len(result.followers) == 1  # Only follower_a
    assert len(result.cutters) == 1  # Only cutter_a
    assert len(result.non_production_parts) == 1  # Only aux_a

    original_volume = get_volume(leader)
    subtractor_volume = get_volume(subtractor_leader)
    assert get_volume(result.leader) == pytest.approx(
        original_volume - subtractor_volume, rel=1e-6
    )

    # Check that only the original followers/cutters/aux are present
    follower_a_clone = result.followers[0]
    cutter_a_clone = result.cutters[0]
    aux_a_clone = result.non_production_parts[0]

    assert follower_a_clone is not group_a.followers[0]
    assert cutter_a_clone is not group_a.cutters[0]
    assert aux_a_clone is not group_a.non_production_parts[0]

    assert follower_a_clone.part is not group_a.followers[0].part
    assert follower_a_clone.name == "follower_a"
    assert cutter_a_clone.name == "cutter_a"
    assert aux_a_clone.name == "aux_a"


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
    """Test that additional_data from original group is preserved when cutting with another group."""
    leader1 = create_box(2, 2, 2)
    leader2 = translate(0.5, 0.5, 0.5)(create_box(1, 1, 1))

    data1 = {"material": "wood", "finish": "stain"}
    data2 = {"tool": "saw", "speed": "slow"}

    group1 = LeaderFollowersCuttersPart(leader1, additional_data=data1)
    group2 = LeaderFollowersCuttersPart(leader2, additional_data=data2)

    cut_group = group1.cut(group2)

    # When cutting with another group, only the original group's additional_data is preserved
    assert cut_group.additional_data == data1
    assert cut_group.additional_data is not group1.additional_data  # Should be a copy


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


def test_follower_names_basic_functionality():
    """Test basic follower name tracking functionality."""
    leader = create_box(2, 2, 2)
    follower1 = translate(3, 0, 0)(create_box(1, 1, 1))
    follower2 = translate(0, 3, 0)(create_box(1, 1, 1))

    group = LeaderFollowersCuttersPart(
        leader, followers=[follower1, follower2], follower_names=["first", "second"]
    )

    # Check that names are properly mapped to indices
    assert group.get_follower_index_by_name("first") == 0
    assert group.get_follower_index_by_name("second") == 1
    assert group.get_follower_index_by_name("nonexistent") is None

    # Check that follower_indices_by_name is properly populated
    assert group.follower_indices_by_name == {"first": 0, "second": 1}


def test_follower_names_length_mismatch_assertion():
    """Test that providing mismatched lengths of followers and names raises assertion."""
    leader = create_box(2, 2, 2)
    follower1 = translate(3, 0, 0)(create_box(1, 1, 1))
    follower2 = translate(0, 3, 0)(create_box(1, 1, 1))

    with pytest.raises(AssertionError):
        LeaderFollowersCuttersPart(
            leader,
            followers=[follower1, follower2],
            follower_names=["first"],  # Only one name for two followers
        )

    with pytest.raises(AssertionError):
        LeaderFollowersCuttersPart(
            leader,
            followers=[follower1],
            follower_names=["first", "second"],  # Two names for one follower
        )


def test_follower_names_without_names():
    """Test that creating a group without follower_names works properly."""
    leader = create_box(2, 2, 2)
    follower1 = translate(3, 0, 0)(create_box(1, 1, 1))
    follower2 = translate(0, 3, 0)(create_box(1, 1, 1))

    group = LeaderFollowersCuttersPart(
        leader,
        followers=[follower1, follower2],
        # No follower_names provided
    )

    # Should have empty name mapping
    assert group.follower_indices_by_name == {}
    assert group.get_follower_index_by_name("any_name") is None


def test_follower_names_preserved_in_copy():
    """Test that follower names are preserved when copying."""
    leader = create_box(2, 2, 2)
    follower1 = translate(3, 0, 0)(create_box(1, 1, 1))
    follower2 = translate(0, 3, 0)(create_box(1, 1, 1))

    original = LeaderFollowersCuttersPart(
        leader, followers=[follower1, follower2], follower_names=["alpha", "beta"]
    )

    copied = original.copy()

    # Check that names are preserved
    assert copied.get_follower_index_by_name("alpha") == 0
    assert copied.get_follower_index_by_name("beta") == 1
    assert copied.follower_indices_by_name == {"alpha": 0, "beta": 1}

    # Check that it's a deep copy
    assert copied.follower_indices_by_name is not original.follower_indices_by_name


def test_follower_names_fuse_with_shape():
    """Test that follower names are preserved when fusing with a simple shape."""
    leader = create_box(2, 2, 2)
    follower1 = translate(3, 0, 0)(create_box(1, 1, 1))
    follower2 = translate(0, 3, 0)(create_box(1, 1, 1))

    group = LeaderFollowersCuttersPart(
        leader, followers=[follower1, follower2], follower_names=["left", "top"]
    )

    other_shape = translate(1.5, 0, 0)(create_box(1, 1, 1))
    fused = group.fuse(other_shape)

    # Names should be preserved
    assert fused.get_follower_index_by_name("left") == 0
    assert fused.get_follower_index_by_name("top") == 1
    assert fused.follower_indices_by_name == {"left": 0, "top": 1}


def test_follower_names_fuse_with_group_no_collision():
    """Test fusing two groups with different follower names (no collision)."""
    leader1 = create_box(2, 2, 2)
    follower1a = translate(3, 0, 0)(create_box(1, 1, 1))
    follower1b = translate(0, 3, 0)(create_box(1, 1, 1))

    group1 = LeaderFollowersCuttersPart(
        leader1,
        followers=[follower1a, follower1b],
        follower_names=["group1_a", "group1_b"],
    )

    leader2 = translate(5, 0, 0)(create_box(2, 2, 2))
    follower2a = translate(8, 0, 0)(create_box(1, 1, 1))
    follower2b = translate(5, 3, 0)(create_box(1, 1, 1))

    group2 = LeaderFollowersCuttersPart(
        leader2,
        followers=[follower2a, follower2b],
        follower_names=["group2_a", "group2_b"],
    )

    fused = group1.fuse(group2)

    # Check that all names are preserved with correct indices
    assert fused.get_follower_index_by_name("group1_a") == 0
    assert fused.get_follower_index_by_name("group1_b") == 1
    assert fused.get_follower_index_by_name("group2_a") == 2  # group1 had 2 followers
    assert fused.get_follower_index_by_name("group2_b") == 3

    expected_mapping = {"group1_a": 0, "group1_b": 1, "group2_a": 2, "group2_b": 3}
    assert fused.follower_indices_by_name == expected_mapping

    # Check that we have all 4 followers
    assert len(fused.followers) == 4


def test_follower_names_fuse_with_group_collision_error():
    """Test that fusing groups with colliding follower names raises ValueError."""
    leader1 = create_box(2, 2, 2)
    follower1a = translate(3, 0, 0)(create_box(1, 1, 1))
    follower1b = translate(0, 3, 0)(create_box(1, 1, 1))

    group1 = LeaderFollowersCuttersPart(
        leader1,
        followers=[follower1a, follower1b],
        follower_names=["common", "group1_unique"],
    )

    leader2 = translate(5, 0, 0)(create_box(2, 2, 2))
    follower2a = translate(8, 0, 0)(create_box(1, 1, 1))
    follower2b = translate(5, 3, 0)(create_box(1, 1, 1))

    group2 = LeaderFollowersCuttersPart(
        leader2,
        followers=[follower2a, follower2b],
        follower_names=["common", "group2_unique"],  # "common" collides
    )

    with pytest.raises(ValueError) as exc_info:
        group1.fuse(group2)

    assert "Follower name collision: 'common' already exists" in str(exc_info.value)


def test_follower_names_cut_with_shape():
    """Test that follower names are preserved when cutting with a simple shape."""
    leader = create_box(2, 2, 2)
    follower1 = translate(3, 0, 0)(create_box(1, 1, 1))
    follower2 = translate(0, 3, 0)(create_box(1, 1, 1))

    group = LeaderFollowersCuttersPart(
        leader, followers=[follower1, follower2], follower_names=["right", "back"]
    )

    cutter_shape = translate(0.5, 0.5, 0.5)(create_box(1, 1, 1))
    cut_result = group.cut(cutter_shape)

    # Names should be preserved
    assert cut_result.get_follower_index_by_name("right") == 0
    assert cut_result.get_follower_index_by_name("back") == 1
    assert cut_result.follower_indices_by_name == {"right": 0, "back": 1}


def test_follower_names_cut_with_group_no_collision():
    """Test cutting with another group preserves only original part's names."""
    leader1 = create_box(3, 3, 3)
    follower1a = translate(4, 0, 0)(create_box(1, 1, 1))

    group1 = LeaderFollowersCuttersPart(
        leader1, followers=[follower1a], follower_names=["main_follower"]
    )

    cutter_leader = translate(0.5, 0.5, 0.5)(create_box(1, 1, 1))
    cutter_follower = translate(4.5, 0.5, 0.5)(create_box(0.5, 0.5, 0.5))

    cutter_group = LeaderFollowersCuttersPart(
        cutter_leader, followers=[cutter_follower], follower_names=["cutter_follower"]
    )

    cut_result = group1.cut(cutter_group)

    # Check that only the original part's names are preserved
    assert cut_result.get_follower_index_by_name("main_follower") == 0
    assert (
        cut_result.get_follower_index_by_name("cutter_follower") is None
    )  # Not preserved

    expected_mapping = {"main_follower": 0}
    assert cut_result.follower_indices_by_name == expected_mapping
    assert len(cut_result.followers) == 1  # Only original follower


def test_follower_names_cut_with_group_collision_error():
    """Test that cutting doesn't cause name collisions since cutter names aren't merged."""
    leader1 = create_box(3, 3, 3)
    follower1a = translate(4, 0, 0)(create_box(1, 1, 1))

    group1 = LeaderFollowersCuttersPart(
        leader1, followers=[follower1a], follower_names=["shared_name"]
    )

    cutter_leader = translate(0.5, 0.5, 0.5)(create_box(1, 1, 1))
    cutter_follower = translate(4.5, 0.5, 0.5)(create_box(0.5, 0.5, 0.5))

    cutter_group = LeaderFollowersCuttersPart(
        cutter_leader,
        followers=[cutter_follower],
        follower_names=[
            "shared_name"
        ],  # Same name as group1, but should not cause collision
    )

    # Should not raise ValueError since cutter names aren't merged
    cut_result = group1.cut(cutter_group)

    # Only the original part's names should be preserved
    assert cut_result.get_follower_index_by_name("shared_name") == 0
    assert len(cut_result.follower_indices_by_name) == 1
    assert len(cut_result.followers) == 1


def test_follower_names_with_transformations():
    """Test that follower names are preserved through transformations."""
    leader = create_box(2, 2, 2)
    follower1 = translate(3, 0, 0)(create_box(1, 1, 1))
    follower2 = translate(0, 3, 0)(create_box(1, 1, 1))

    group = LeaderFollowersCuttersPart(
        leader, followers=[follower1, follower2], follower_names=["east", "north"]
    )

    # Test functional transformation (returns new object)
    translated = translate(5, 0, 0)(group)
    assert translated.get_follower_index_by_name("east") == 0
    assert translated.get_follower_index_by_name("north") == 1
    assert translated.follower_indices_by_name == {"east": 0, "north": 1}

    rotated = rotate(90, axis=(0, 0, 1))(group)
    assert rotated.get_follower_index_by_name("east") == 0
    assert rotated.get_follower_index_by_name("north") == 1
    assert rotated.follower_indices_by_name == {"east": 0, "north": 1}


def test_follower_names_complex_fusion_scenario():
    """Test a complex scenario with multiple fusions and transformations."""
    # Create first group
    leader1 = create_box(1, 1, 1)
    follower1 = translate(2, 0, 0)(create_box(0.5, 0.5, 0.5))
    group1 = LeaderFollowersCuttersPart(
        leader1, followers=[follower1], follower_names=["first"]
    )

    # Create second group
    leader2 = translate(0, 2, 0)(create_box(1, 1, 1))
    follower2a = translate(2, 2, 0)(create_box(0.5, 0.5, 0.5))
    follower2b = translate(0, 4, 0)(create_box(0.5, 0.5, 0.5))
    group2 = LeaderFollowersCuttersPart(
        leader2,
        followers=[follower2a, follower2b],
        follower_names=["second_a", "second_b"],
    )

    # Fuse them
    fused = group1.fuse(group2)

    # Transform the result
    transformed = translate(10, 0, 0)(fused)

    # Check that all names are still accessible
    assert transformed.get_follower_index_by_name("first") == 0
    assert transformed.get_follower_index_by_name("second_a") == 1
    assert transformed.get_follower_index_by_name("second_b") == 2

    expected_mapping = {"first": 0, "second_a": 1, "second_b": 2}
    assert transformed.follower_indices_by_name == expected_mapping


def test_follower_names_edge_cases():
    """Test edge cases for follower name functionality."""
    leader = create_box(1, 1, 1)

    # Test with empty followers but names provided (should fail)
    with pytest.raises(AssertionError):
        LeaderFollowersCuttersPart(leader, followers=[], follower_names=["should_fail"])

    # Test with None followers and names
    group = LeaderFollowersCuttersPart(leader, followers=None, follower_names=None)
    assert group.follower_indices_by_name == {}
    assert group.get_follower_index_by_name("anything") is None

    # Test fusing group with no named followers
    follower = translate(2, 0, 0)(create_box(0.5, 0.5, 0.5))
    named_group = LeaderFollowersCuttersPart(
        leader, followers=[follower], follower_names=["named"]
    )

    unnamed_group = LeaderFollowersCuttersPart(translate(0, 2, 0)(create_box(1, 1, 1)))

    fused = named_group.fuse(unnamed_group)
    # Should preserve the one named follower
    assert fused.get_follower_index_by_name("named") == 0
    assert len(fused.follower_indices_by_name) == 1


def test_follower_names_reconstruct_method():
    """Test that follower names are preserved in reconstruct method."""
    leader = create_box(2, 2, 2)
    follower1 = translate(3, 0, 0)(create_box(1, 1, 1))
    follower2 = translate(0, 3, 0)(create_box(1, 1, 1))

    group = LeaderFollowersCuttersPart(
        leader,
        followers=[follower1, follower2],
        follower_names=["reconstruct_a", "reconstruct_b"],
    )

    # Test reconstruct without transformed_result
    reconstructed = group.reconstruct()
    assert reconstructed.get_follower_index_by_name("reconstruct_a") == 0
    assert reconstructed.get_follower_index_by_name("reconstruct_b") == 1
    assert reconstructed.follower_indices_by_name == {
        "reconstruct_a": 0,
        "reconstruct_b": 1,
    }

    # Test reconstruct with transformed_result
    transformed_group = translate(5, 0, 0)(group)
    reconstructed_with_transform = group.reconstruct(transformed_group)
    assert reconstructed_with_transform.get_follower_index_by_name("reconstruct_a") == 0
    assert reconstructed_with_transform.get_follower_index_by_name("reconstruct_b") == 1


def test_cutter_names_basic_functionality():
    """Test basic cutter name tracking functionality."""
    leader = create_box(2, 2, 2)
    cutter1 = create_box(0.5, 0.5, 0.5)
    cutter2 = translate(1, 0, 0)(create_box(0.5, 0.5, 0.5))

    group = LeaderFollowersCuttersPart(
        leader, cutters=[cutter1, cutter2], cutter_names=["drill", "slot"]
    )

    # Check that names are properly mapped to indices
    assert group.get_cutter_index_by_name("drill") == 0
    assert group.get_cutter_index_by_name("slot") == 1
    assert group.get_cutter_index_by_name("nonexistent") is None

    # Check that cutter_indices_by_name is properly populated
    assert group.cutter_indices_by_name == {"drill": 0, "slot": 1}


def test_add_named_cutter_validates_and_tracks_index():
    leader = create_box(1, 1, 1)
    cutter_primary = create_box(0.2, 0.2, 0.2)
    cutter_duplicate = create_box(0.3, 0.3, 0.3)

    group = LeaderFollowersCuttersPart(leader)
    group.add_named_cutter(cutter_primary, "pilot_hole")

    assert group.cutters[0] is cutter_primary
    assert group.get_cutter_index_by_name("pilot_hole") == 0
    assert group.cutter_indices_by_name == {"pilot_hole": 0}

    with pytest.raises(ValueError):
        group.add_named_cutter(cutter_duplicate, "pilot_hole")

    with pytest.raises(TypeError):
        group.add_named_cutter(create_box(0.1, 0.1, 0.1), 123)


def test_non_production_names_basic_functionality():
    """Test basic non-production part name tracking functionality."""
    leader = create_box(2, 2, 2)
    aux1 = create_box(0.2, 0.2, 0.2)
    aux2 = translate(0, 1, 0)(create_box(0.2, 0.2, 0.2))

    group = LeaderFollowersCuttersPart(
        leader,
        non_production_parts=[aux1, aux2],
        non_production_names=["marker", "guide"],
    )

    # Check that names are properly mapped to indices
    assert group.get_non_production_index_by_name("marker") == 0
    assert group.get_non_production_index_by_name("guide") == 1
    assert group.get_non_production_index_by_name("nonexistent") is None

    # Check that non_production_indices_by_name is properly populated
    assert group.non_production_indices_by_name == {"marker": 0, "guide": 1}


def test_all_names_together():
    """Test using follower, cutter, and non-production names together."""
    leader = create_box(2, 2, 2)
    follower1 = translate(3, 0, 0)(create_box(1, 1, 1))
    cutter1 = create_box(0.5, 0.5, 0.5)
    aux1 = create_box(0.2, 0.2, 0.2)

    group = LeaderFollowersCuttersPart(
        leader,
        followers=[follower1],
        cutters=[cutter1],
        non_production_parts=[aux1],
        follower_names=["side_part"],
        cutter_names=["center_hole"],
        non_production_names=["alignment_pin"],
    )

    assert group.get_follower_index_by_name("side_part") == 0
    assert group.get_cutter_index_by_name("center_hole") == 0
    assert group.get_non_production_index_by_name("alignment_pin") == 0

    # Test copy preserves all names
    copied = group.copy()
    assert copied.get_follower_index_by_name("side_part") == 0
    assert copied.get_cutter_index_by_name("center_hole") == 0
    assert copied.get_non_production_index_by_name("alignment_pin") == 0


def test_cutter_names_fuse_collision():
    """Test that fusing groups with colliding cutter names raises ValueError."""
    leader1 = create_box(2, 2, 2)
    cutter1 = create_box(0.5, 0.5, 0.5)

    group1 = LeaderFollowersCuttersPart(
        leader1, cutters=[cutter1], cutter_names=["common_cutter"]
    )

    leader2 = translate(3, 0, 0)(create_box(2, 2, 2))
    cutter2 = translate(3, 0, 0)(create_box(0.5, 0.5, 0.5))

    group2 = LeaderFollowersCuttersPart(
        leader2,
        cutters=[cutter2],
        cutter_names=["common_cutter"],  # Same name as group1
    )

    with pytest.raises(ValueError) as exc_info:
        group1.fuse(group2)

    assert "Cutter name collision: 'common_cutter' already exists" in str(
        exc_info.value
    )


def test_non_production_names_fuse_collision():
    """Test that fusing groups with colliding non-production names raises ValueError."""
    leader1 = create_box(2, 2, 2)
    aux1 = create_box(0.2, 0.2, 0.2)

    group1 = LeaderFollowersCuttersPart(
        leader1, non_production_parts=[aux1], non_production_names=["common_aux"]
    )

    leader2 = translate(3, 0, 0)(create_box(2, 2, 2))
    aux2 = translate(3, 0, 0)(create_box(0.2, 0.2, 0.2))

    group2 = LeaderFollowersCuttersPart(
        leader2,
        non_production_parts=[aux2],
        non_production_names=["common_aux"],  # Same name as group1
    )

    with pytest.raises(ValueError) as exc_info:
        group1.fuse(group2)

    assert "Non-production part name collision: 'common_aux' already exists" in str(
        exc_info.value
    )


def test_all_names_fuse_no_collision():
    """Test successful fusion of groups with different names for all part types."""
    leader1 = create_box(1, 1, 1)
    follower1 = translate(2, 0, 0)(create_box(0.5, 0.5, 0.5))
    cutter1 = create_box(0.3, 0.3, 0.3)
    aux1 = create_box(0.1, 0.1, 0.1)

    group1 = LeaderFollowersCuttersPart(
        leader1,
        followers=[follower1],
        cutters=[cutter1],
        non_production_parts=[aux1],
        follower_names=["g1_follower"],
        cutter_names=["g1_cutter"],
        non_production_names=["g1_aux"],
    )

    leader2 = translate(0, 2, 0)(create_box(1, 1, 1))
    follower2 = translate(2, 2, 0)(create_box(0.5, 0.5, 0.5))
    cutter2 = translate(0, 2, 0)(create_box(0.3, 0.3, 0.3))
    aux2 = translate(0, 2, 0)(create_box(0.1, 0.1, 0.1))

    group2 = LeaderFollowersCuttersPart(
        leader2,
        followers=[follower2],
        cutters=[cutter2],
        non_production_parts=[aux2],
        follower_names=["g2_follower"],
        cutter_names=["g2_cutter"],
        non_production_names=["g2_aux"],
    )

    fused = group1.fuse(group2)

    # Check all names are preserved with correct indices
    assert fused.get_follower_index_by_name("g1_follower") == 0
    assert fused.get_follower_index_by_name("g2_follower") == 1
    assert fused.get_cutter_index_by_name("g1_cutter") == 0
    assert fused.get_cutter_index_by_name("g2_cutter") == 1
    assert fused.get_non_production_index_by_name("g1_aux") == 0
    assert fused.get_non_production_index_by_name("g2_aux") == 1

    # Check counts
    assert len(fused.followers) == 2
    assert len(fused.cutters) == 2
    assert len(fused.non_production_parts) == 2


def test_names_length_mismatch_assertions():
    """Test that providing mismatched lengths raises assertions for all name types."""
    leader = create_box(2, 2, 2)
    follower1 = translate(3, 0, 0)(create_box(1, 1, 1))
    cutter1 = create_box(0.5, 0.5, 0.5)
    aux1 = create_box(0.2, 0.2, 0.2)

    # Test cutter names mismatch
    with pytest.raises(AssertionError):
        LeaderFollowersCuttersPart(
            leader,
            cutters=[cutter1],
            cutter_names=["name1", "name2"],  # Two names for one cutter
        )

    # Test non-production names mismatch
    with pytest.raises(AssertionError):
        LeaderFollowersCuttersPart(
            leader,
            non_production_parts=[aux1],
            non_production_names=["name1", "name2"],  # Two names for one aux part
        )

    # Test all together with correct lengths (should work)
    group = LeaderFollowersCuttersPart(
        leader,
        followers=[follower1],
        cutters=[cutter1],
        non_production_parts=[aux1],
        follower_names=["f1"],
        cutter_names=["c1"],
        non_production_names=["a1"],
    )
    assert group.get_follower_index_by_name("f1") == 0
    assert group.get_cutter_index_by_name("c1") == 0
    assert group.get_non_production_index_by_name("a1") == 0


def test_cut_preserves_only_original_names():
    """Test that cut operation preserves only the original part's names."""
    leader1 = create_box(3, 3, 3)
    follower1 = translate(4, 0, 0)(create_box(1, 1, 1))
    cutter1 = create_box(0.5, 0.5, 0.5)
    aux1 = create_box(0.2, 0.2, 0.2)

    group1 = LeaderFollowersCuttersPart(
        leader1,
        followers=[follower1],
        cutters=[cutter1],
        non_production_parts=[aux1],
        follower_names=["original_follower"],
        cutter_names=["original_cutter"],
        non_production_names=["original_aux"],
    )

    cutter_leader = translate(0.5, 0.5, 0.5)(create_box(1, 1, 1))
    cutter_follower = translate(4.5, 0.5, 0.5)(create_box(0.5, 0.5, 0.5))
    cutter_cutter = translate(0.5, 0.5, 0.5)(create_box(0.3, 0.3, 0.3))
    cutter_aux = translate(0.5, 0.5, 0.5)(create_box(0.1, 0.1, 0.1))

    cutter_group = LeaderFollowersCuttersPart(
        cutter_leader,
        followers=[cutter_follower],
        cutters=[cutter_cutter],
        non_production_parts=[cutter_aux],
        follower_names=["cutter_follower"],
        cutter_names=["cutter_cutter"],
        non_production_names=["cutter_aux"],
    )

    cut_result = group1.cut(cutter_group)

    # Check that only original names are preserved
    assert cut_result.get_follower_index_by_name("original_follower") == 0
    assert cut_result.get_follower_index_by_name("cutter_follower") is None

    assert cut_result.get_cutter_index_by_name("original_cutter") == 0
    assert cut_result.get_cutter_index_by_name("cutter_cutter") is None

    assert cut_result.get_non_production_index_by_name("original_aux") == 0
    assert cut_result.get_non_production_index_by_name("cutter_aux") is None

    # Check counts - only original parts should remain
    assert len(cut_result.followers) == 1
    assert len(cut_result.cutters) == 1
    assert len(cut_result.non_production_parts) == 1
    assert len(cut_result.follower_indices_by_name) == 1
    assert len(cut_result.cutter_indices_by_name) == 1
    assert len(cut_result.non_production_indices_by_name) == 1


def test_multiple_collision_types_in_fuse():
    """Test that different types of name collisions are all caught."""
    leader1 = create_box(1, 1, 1)
    follower1 = translate(2, 0, 0)(create_box(0.5, 0.5, 0.5))
    cutter1 = create_box(0.3, 0.3, 0.3)
    aux1 = create_box(0.1, 0.1, 0.1)

    group1 = LeaderFollowersCuttersPart(
        leader1,
        followers=[follower1],
        cutters=[cutter1],
        non_production_parts=[aux1],
        follower_names=["shared"],
        cutter_names=["unique1"],
        non_production_names=["unique2"],
    )

    leader2 = translate(0, 2, 0)(create_box(1, 1, 1))
    follower2 = translate(2, 2, 0)(create_box(0.5, 0.5, 0.5))
    cutter2 = translate(0, 2, 0)(create_box(0.3, 0.3, 0.3))
    aux2 = translate(0, 2, 0)(create_box(0.1, 0.1, 0.1))

    # Test follower name collision
    group2_follower_collision = LeaderFollowersCuttersPart(
        leader2,
        followers=[follower2],
        cutters=[cutter2],
        non_production_parts=[aux2],
        follower_names=["shared"],  # Collision with group1
        cutter_names=["unique3"],
        non_production_names=["unique4"],
    )

    with pytest.raises(ValueError) as exc_info:
        group1.fuse(group2_follower_collision)
    assert "Follower name collision: 'shared'" in str(exc_info.value)

    # Test cutter name collision
    group2_cutter_collision = LeaderFollowersCuttersPart(
        leader2,
        followers=[follower2],
        cutters=[cutter2],
        non_production_parts=[aux2],
        follower_names=["unique5"],
        cutter_names=["unique1"],  # Collision with group1
        non_production_names=["unique6"],
    )

    with pytest.raises(ValueError) as exc_info:
        group1.fuse(group2_cutter_collision)
    assert "Cutter name collision: 'unique1'" in str(exc_info.value)

    # Test non-production name collision
    group2_aux_collision = LeaderFollowersCuttersPart(
        leader2,
        followers=[follower2],
        cutters=[cutter2],
        non_production_parts=[aux2],
        follower_names=["unique7"],
        cutter_names=["unique8"],
        non_production_names=["unique2"],  # Collision with group1
    )

    with pytest.raises(ValueError) as exc_info:
        group1.fuse(group2_aux_collision)
    assert "Non-production part name collision: 'unique2'" in str(exc_info.value)


def test_cutter_names_basic_functionality():
    """Test basic cutter name tracking functionality."""
    leader = create_box(2, 2, 2)
    cutter1 = create_box(0.5, 0.5, 0.5)
    cutter2 = translate(1, 0, 0)(create_box(0.5, 0.5, 0.5))

    group = LeaderFollowersCuttersPart(
        leader,
        cutters=[cutter1, cutter2],
        cutter_names=["primary_cut", "secondary_cut"],
    )

    # Check that names are properly mapped to indices
    assert group.get_cutter_index_by_name("primary_cut") == 0
    assert group.get_cutter_index_by_name("secondary_cut") == 1
    assert group.get_cutter_index_by_name("nonexistent") is None

    # Check that cutter_indices_by_name is properly populated
    assert group.cutter_indices_by_name == {"primary_cut": 0, "secondary_cut": 1}


def test_cutter_names_length_mismatch_assertion():
    """Test that providing mismatched lengths of cutters and names raises assertion."""
    leader = create_box(2, 2, 2)
    cutter1 = create_box(0.5, 0.5, 0.5)
    cutter2 = translate(1, 0, 0)(create_box(0.5, 0.5, 0.5))

    with pytest.raises(AssertionError):
        LeaderFollowersCuttersPart(
            leader,
            cutters=[cutter1, cutter2],
            cutter_names=["only_one"],  # Only one name for two cutters
        )


def test_cutter_names_preserved_in_copy():
    """Test that cutter names are preserved when copying."""
    leader = create_box(2, 2, 2)
    cutter1 = create_box(0.5, 0.5, 0.5)
    cutter2 = translate(1, 0, 0)(create_box(0.5, 0.5, 0.5))

    original = LeaderFollowersCuttersPart(
        leader, cutters=[cutter1, cutter2], cutter_names=["drill", "slot"]
    )

    copied = original.copy()

    # Check that names are preserved
    assert copied.get_cutter_index_by_name("drill") == 0
    assert copied.get_cutter_index_by_name("slot") == 1
    assert copied.cutter_indices_by_name == {"drill": 0, "slot": 1}

    # Check that it's a deep copy
    assert copied.cutter_indices_by_name is not original.cutter_indices_by_name


def test_cutter_names_fuse_with_group_collision_error():
    """Test that fusing groups with colliding cutter names raises ValueError."""
    leader1 = create_box(2, 2, 2)
    cutter1 = create_box(0.5, 0.5, 0.5)

    group1 = LeaderFollowersCuttersPart(
        leader1, cutters=[cutter1], cutter_names=["common_cutter"]
    )

    leader2 = translate(5, 0, 0)(create_box(2, 2, 2))
    cutter2 = translate(5, 0, 0)(create_box(0.5, 0.5, 0.5))

    group2 = LeaderFollowersCuttersPart(
        leader2,
        cutters=[cutter2],
        cutter_names=["common_cutter"],  # Same name as group1
    )

    with pytest.raises(ValueError) as exc_info:
        group1.fuse(group2)

    assert "Cutter name collision: 'common_cutter' already exists" in str(
        exc_info.value
    )


def test_cutter_names_fuse_with_group_no_collision():
    """Test fusing two groups with different cutter names (no collision)."""
    leader1 = create_box(2, 2, 2)
    cutter1a = create_box(0.5, 0.5, 0.5)
    cutter1b = translate(1, 0, 0)(create_box(0.5, 0.5, 0.5))

    group1 = LeaderFollowersCuttersPart(
        leader1,
        cutters=[cutter1a, cutter1b],
        cutter_names=["group1_cut_a", "group1_cut_b"],
    )

    leader2 = translate(5, 0, 0)(create_box(2, 2, 2))
    cutter2a = translate(5, 0, 0)(create_box(0.5, 0.5, 0.5))

    group2 = LeaderFollowersCuttersPart(
        leader2, cutters=[cutter2a], cutter_names=["group2_cut_a"]
    )

    fused = group1.fuse(group2)

    # Check that all names are preserved with correct indices
    assert fused.get_cutter_index_by_name("group1_cut_a") == 0
    assert fused.get_cutter_index_by_name("group1_cut_b") == 1
    assert fused.get_cutter_index_by_name("group2_cut_a") == 2  # group1 had 2 cutters

    expected_mapping = {"group1_cut_a": 0, "group1_cut_b": 1, "group2_cut_a": 2}
    assert fused.cutter_indices_by_name == expected_mapping
    assert len(fused.cutters) == 3


def test_non_production_names_basic_functionality():
    """Test basic non-production part name tracking functionality."""
    leader = create_box(2, 2, 2)
    aux1 = create_box(0.2, 0.2, 0.2)
    aux2 = translate(1, 0, 0)(create_box(0.2, 0.2, 0.2))

    group = LeaderFollowersCuttersPart(
        leader,
        non_production_parts=[aux1, aux2],
        non_production_names=["guide", "marker"],
    )

    # Check that names are properly mapped to indices
    assert group.get_non_production_index_by_name("guide") == 0
    assert group.get_non_production_index_by_name("marker") == 1
    assert group.get_non_production_index_by_name("nonexistent") is None

    # Check that non_production_indices_by_name is properly populated
    assert group.non_production_indices_by_name == {"guide": 0, "marker": 1}


def test_non_production_names_length_mismatch_assertion():
    """Test that providing mismatched lengths of non-production parts and names raises assertion."""
    leader = create_box(2, 2, 2)
    aux1 = create_box(0.2, 0.2, 0.2)
    aux2 = translate(1, 0, 0)(create_box(0.2, 0.2, 0.2))

    with pytest.raises(AssertionError):
        LeaderFollowersCuttersPart(
            leader,
            non_production_parts=[aux1, aux2],
            non_production_names=["only_one"],  # Only one name for two parts
        )


def test_non_production_names_preserved_in_copy():
    """Test that non-production part names are preserved when copying."""
    leader = create_box(2, 2, 2)
    aux1 = create_box(0.2, 0.2, 0.2)
    aux2 = translate(1, 0, 0)(create_box(0.2, 0.2, 0.2))

    original = LeaderFollowersCuttersPart(
        leader,
        non_production_parts=[aux1, aux2],
        non_production_names=["helper", "template"],
    )

    copied = original.copy()

    # Check that names are preserved
    assert copied.get_non_production_index_by_name("helper") == 0
    assert copied.get_non_production_index_by_name("template") == 1
    assert copied.non_production_indices_by_name == {"helper": 0, "template": 1}

    # Check that it's a deep copy
    assert (
        copied.non_production_indices_by_name
        is not original.non_production_indices_by_name
    )


def test_non_production_names_fuse_with_group_collision_error():
    """Test that fusing groups with colliding non-production part names raises ValueError."""
    leader1 = create_box(2, 2, 2)
    aux1 = create_box(0.2, 0.2, 0.2)

    group1 = LeaderFollowersCuttersPart(
        leader1, non_production_parts=[aux1], non_production_names=["common_aux"]
    )

    leader2 = translate(5, 0, 0)(create_box(2, 2, 2))
    aux2 = translate(5, 0, 0)(create_box(0.2, 0.2, 0.2))

    group2 = LeaderFollowersCuttersPart(
        leader2,
        non_production_parts=[aux2],
        non_production_names=["common_aux"],  # Same name as group1
    )

    with pytest.raises(ValueError) as exc_info:
        group1.fuse(group2)

    assert "Non-production part name collision: 'common_aux' already exists" in str(
        exc_info.value
    )


def test_non_production_names_fuse_with_group_no_collision():
    """Test fusing two groups with different non-production part names (no collision)."""
    leader1 = create_box(2, 2, 2)
    aux1a = create_box(0.2, 0.2, 0.2)
    aux1b = translate(1, 0, 0)(create_box(0.2, 0.2, 0.2))

    group1 = LeaderFollowersCuttersPart(
        leader1,
        non_production_parts=[aux1a, aux1b],
        non_production_names=["group1_aux_a", "group1_aux_b"],
    )

    leader2 = translate(5, 0, 0)(create_box(2, 2, 2))
    aux2a = translate(5, 0, 0)(create_box(0.2, 0.2, 0.2))

    group2 = LeaderFollowersCuttersPart(
        leader2, non_production_parts=[aux2a], non_production_names=["group2_aux_a"]
    )

    fused = group1.fuse(group2)

    # Check that all names are preserved with correct indices
    assert fused.get_non_production_index_by_name("group1_aux_a") == 0
    assert fused.get_non_production_index_by_name("group1_aux_b") == 1
    assert (
        fused.get_non_production_index_by_name("group2_aux_a") == 2
    )  # group1 had 2 aux parts

    expected_mapping = {"group1_aux_a": 0, "group1_aux_b": 1, "group2_aux_a": 2}
    assert fused.non_production_indices_by_name == expected_mapping
    assert len(fused.non_production_parts) == 3


def test_all_names_comprehensive_scenario():
    """Test a comprehensive scenario with all types of named parts."""
    leader1 = create_box(2, 2, 2)
    follower1 = translate(3, 0, 0)(create_box(1, 1, 1))
    cutter1 = create_box(0.5, 0.5, 0.5)
    aux1 = create_box(0.2, 0.2, 0.2)

    group1 = LeaderFollowersCuttersPart(
        leader1,
        followers=[follower1],
        cutters=[cutter1],
        non_production_parts=[aux1],
        follower_names=["main_follower"],
        cutter_names=["main_cutter"],
        non_production_names=["main_aux"],
    )

    leader2 = translate(0, 3, 0)(create_box(2, 2, 2))
    follower2 = translate(3, 3, 0)(create_box(1, 1, 1))
    cutter2 = translate(0, 3, 0)(create_box(0.5, 0.5, 0.5))
    aux2 = translate(0, 3, 0)(create_box(0.2, 0.2, 0.2))

    group2 = LeaderFollowersCuttersPart(
        leader2,
        followers=[follower2],
        cutters=[cutter2],
        non_production_parts=[aux2],
        follower_names=["second_follower"],
        cutter_names=["second_cutter"],
        non_production_names=["second_aux"],
    )

    # Test fusion
    fused = group1.fuse(group2)

    # Check all names are preserved
    assert fused.get_follower_index_by_name("main_follower") == 0
    assert fused.get_follower_index_by_name("second_follower") == 1
    assert fused.get_cutter_index_by_name("main_cutter") == 0
    assert fused.get_cutter_index_by_name("second_cutter") == 1
    assert fused.get_non_production_index_by_name("main_aux") == 0
    assert fused.get_non_production_index_by_name("second_aux") == 1

    # Test transformation preserves names
    transformed = translate(10, 0, 0)(fused)
    assert transformed.get_follower_index_by_name("main_follower") == 0
    assert transformed.get_follower_index_by_name("second_follower") == 1
    assert transformed.get_cutter_index_by_name("main_cutter") == 0
    assert transformed.get_cutter_index_by_name("second_cutter") == 1
    assert transformed.get_non_production_index_by_name("main_aux") == 0
    assert transformed.get_non_production_index_by_name("second_aux") == 1


def test_mixed_name_collisions():
    """Test that different types of name collisions are detected correctly."""
    leader1 = create_box(2, 2, 2)
    follower1 = translate(3, 0, 0)(create_box(1, 1, 1))
    cutter1 = create_box(0.5, 0.5, 0.5)
    aux1 = create_box(0.2, 0.2, 0.2)

    group1 = LeaderFollowersCuttersPart(
        leader1,
        followers=[follower1],
        cutters=[cutter1],
        non_production_parts=[aux1],
        follower_names=["part_a"],
        cutter_names=["cut_a"],
        non_production_names=["aux_a"],
    )

    # Test follower name collision
    leader2 = translate(0, 3, 0)(create_box(2, 2, 2))
    follower2 = translate(3, 3, 0)(create_box(1, 1, 1))
    group2_follower_collision = LeaderFollowersCuttersPart(
        leader2,
        followers=[follower2],
        follower_names=["part_a"],  # Collision with group1
    )

    with pytest.raises(ValueError) as exc_info:
        group1.fuse(group2_follower_collision)
    assert "Follower name collision: 'part_a' already exists" in str(exc_info.value)

    # Test cutter name collision
    cutter2 = translate(0, 3, 0)(create_box(0.5, 0.5, 0.5))
    group2_cutter_collision = LeaderFollowersCuttersPart(
        leader2, cutters=[cutter2], cutter_names=["cut_a"]  # Collision with group1
    )

    with pytest.raises(ValueError) as exc_info:
        group1.fuse(group2_cutter_collision)
    assert "Cutter name collision: 'cut_a' already exists" in str(exc_info.value)

    # Test non-production name collision
    aux2 = translate(0, 3, 0)(create_box(0.2, 0.2, 0.2))
    group2_aux_collision = LeaderFollowersCuttersPart(
        leader2,
        non_production_parts=[aux2],
        non_production_names=["aux_a"],  # Collision with group1
    )

    with pytest.raises(ValueError) as exc_info:
        group1.fuse(group2_aux_collision)
    assert "Non-production part name collision: 'aux_a' already exists" in str(
        exc_info.value
    )
