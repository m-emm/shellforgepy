import pytest
from shellforgepy.adapters._adapter import create_box
from shellforgepy.construct.alignment_operations import translate
from shellforgepy.construct.leader_followers_cutters_part import (
    LeaderFollowersCuttersPart,
)


def _assembly(reference_name, *, offset=0.0, hidden=True):
    leader = create_box(2, 2, 2)
    reference = translate(offset + 3, 0, 0)(create_box(1, 1, 1))
    return LeaderFollowersCuttersPart(
        leader,
        non_production_parts=[reference],
        non_production_names=[reference_name],
        hidden_by_default_names=[reference_name] if hidden else None,
    )


def test_hidden_by_default_requires_unique_named_non_production_parts():
    assembly = _assembly("datum", hidden=False)

    assembly.set_hidden_by_default("datum")
    assembly.set_hidden_by_default("datum")
    assert assembly.hidden_by_default_names == ["datum"]

    assembly.set_hidden_by_default("datum", hidden=False)
    assert assembly.hidden_by_default_names == []

    with pytest.raises(KeyError, match="named non-production part"):
        assembly.set_hidden_by_default("missing")
    with pytest.raises(TypeError, match="must be a string"):
        assembly.set_hidden_by_default(42)
    with pytest.raises(TypeError, match="must be a boolean"):
        assembly.set_hidden_by_default("datum", hidden=1)

    other_categories = LeaderFollowersCuttersPart(
        create_box(2, 2, 2),
        followers=[create_box(1, 1, 1)],
        cutters=[create_box(1, 1, 1)],
        follower_names=["follower"],
        cutter_names=["cutter"],
    )
    with pytest.raises(KeyError, match="named non-production part"):
        other_categories.set_hidden_by_default("follower")
    with pytest.raises(KeyError, match="named non-production part"):
        other_categories.set_hidden_by_default("cutter")

    with pytest.raises(ValueError, match="must not contain duplicates"):
        LeaderFollowersCuttersPart(
            create_box(2, 2, 2),
            non_production_parts=[create_box(1, 1, 1)],
            non_production_names=["datum"],
            hidden_by_default_names=["datum", "datum"],
        )


def test_hidden_by_default_follows_copy_prefix_rename_and_reconstruct():
    assembly = _assembly("datum")

    copied = assembly.copy()
    copied.rename_non_production_part("datum", "renamed_datum")
    prefixed = assembly.prefixed_copy("left")
    reconstructed = assembly.reconstruct(assembly.copy())
    transformed = translate(1, 2, 3)(assembly)

    assert assembly.hidden_by_default_names == ["datum"]
    assert copied.hidden_by_default_names == ["renamed_datum"]
    assert prefixed.hidden_by_default_names == ["left_datum"]
    assert reconstructed.hidden_by_default_names == ["datum"]
    assert transformed.hidden_by_default_names == ["datum"]


def test_hidden_by_default_merges_and_cut_retains_only_left_side():
    left = _assembly("left_datum")
    right = _assembly("right_datum", offset=5.0)

    merged = left.merge_except_leader(right)
    fused = left.fuse(right)
    cut = left.cut(right)

    assert merged.hidden_by_default_names == ["left_datum", "right_datum"]
    assert fused.hidden_by_default_names == ["left_datum", "right_datum"]
    assert cut.hidden_by_default_names == ["left_datum"]


def test_copy_preserves_complete_assembly_state_independently():
    assembly = LeaderFollowersCuttersPart(
        create_box(4, 4, 4),
        followers=[create_box(1, 1, 1)],
        cutters=[create_box(0.5, 0.5, 0.5)],
        non_production_parts=[create_box(2, 2, 2), create_box(3, 3, 3)],
        additional_data={"nested": {"values": [1, 2]}},
        follower_names=["follower"],
        cutter_names=["cutter"],
        non_production_names=["datum", "context"],
        direction_vectors=[(1, 2, 3)],
        direction_vector_names=["normal"],
        hidden_by_default_names=["datum"],
    )

    copied = assembly.copy()

    assert copied is not assembly
    assert copied.leader is not assembly.leader
    assert copied.followers[0] is not assembly.followers[0]
    assert copied.cutters[0] is not assembly.cutters[0]
    assert copied.non_production_parts[0] is not assembly.non_production_parts[0]
    assert copied.follower_indices_by_name == assembly.follower_indices_by_name
    assert copied.cutter_indices_by_name == assembly.cutter_indices_by_name
    assert (
        copied.non_production_indices_by_name == assembly.non_production_indices_by_name
    )
    assert (
        copied.direction_vector_indices_by_name
        == assembly.direction_vector_indices_by_name
    )
    assert copied.direction_vectors == assembly.direction_vectors
    assert copied.additional_data == assembly.additional_data
    assert copied.additional_data is not assembly.additional_data
    assert copied.additional_data["nested"] is not assembly.additional_data["nested"]
    assert copied.hidden_by_default_names == ["datum"]
    assert copied.hidden_by_default_names is not assembly.hidden_by_default_names

    copied.rename_non_production_part("datum", "copied_datum")
    copied.additional_data["nested"]["values"].append(3)

    assert assembly.non_production_indices_by_name == {"datum": 0, "context": 1}
    assert assembly.hidden_by_default_names == ["datum"]
    assert assembly.additional_data == {"nested": {"values": [1, 2]}}


def test_filtered_copy_excludes_all_named_artifact_types_and_rebuilds_indices():
    assembly = LeaderFollowersCuttersPart(
        create_box(4, 4, 4),
        additional_data={"nested": {"values": [1, 2]}},
    )

    assembly.followers.append(create_box(1, 1, 1))
    assembly.add_named_follower(create_box(2, 1, 1), "removed_follower")
    assembly.followers.append(create_box(3, 1, 1))
    assembly.add_named_follower(create_box(4, 1, 1), "kept_follower")

    assembly.cutters.append(create_box(0.1, 0.1, 0.1))
    assembly.add_named_cutter(create_box(0.2, 0.2, 0.2), "removed_cutter")
    assembly.cutters.append(create_box(0.3, 0.3, 0.3))
    assembly.add_named_cutter(create_box(0.4, 0.4, 0.4), "kept_cutter")

    assembly.non_production_parts.append(create_box(1, 2, 1))
    assembly.add_named_non_production_part(
        create_box(1, 3, 1),
        "removed_datum",
    )
    assembly.non_production_parts.append(create_box(1, 4, 1))
    assembly.add_named_non_production_part(
        create_box(1, 5, 1),
        "kept_datum",
    )
    assembly.set_hidden_by_default("removed_datum")
    assembly.set_hidden_by_default("kept_datum")

    assembly.direction_vectors.append((1, 0, 0))
    assembly.add_named_direction_vector((0, 1, 0), "removed_direction")
    assembly.direction_vectors.append((0, 0, 1))
    assembly.add_named_direction_vector((1, 1, 0), "kept_direction")

    filtered = assembly.filtered_copy(
        [
            "removed_follower",
            "removed_cutter",
            "removed_datum",
            "removed_direction",
        ]
    )

    assert filtered.leader is not assembly.leader
    assert filtered.follower_indices_by_name == {"kept_follower": 2}
    assert filtered.cutter_indices_by_name == {"kept_cutter": 2}
    assert filtered.non_production_indices_by_name == {"kept_datum": 2}
    assert filtered.direction_vector_indices_by_name == {"kept_direction": 2}
    assert len(filtered.followers) == 3
    assert len(filtered.cutters) == 3
    assert len(filtered.non_production_parts) == 3
    assert filtered.direction_vectors == [(1, 0, 0), (0, 0, 1), (1, 1, 0)]

    retained_original_parts = (
        [assembly.followers[index] for index in (0, 2, 3)]
        + [assembly.cutters[index] for index in (0, 2, 3)]
        + [assembly.non_production_parts[index] for index in (0, 2, 3)]
    )
    retained_copied_parts = (
        filtered.followers + filtered.cutters + filtered.non_production_parts
    )
    assert all(
        copied_part is not original_part
        for copied_part, original_part in zip(
            retained_copied_parts,
            retained_original_parts,
        )
    )

    assert filtered.hidden_by_default_names == ["kept_datum"]
    assert filtered.hidden_by_default_names is not assembly.hidden_by_default_names
    assert filtered.additional_data == assembly.additional_data
    assert filtered.additional_data is not assembly.additional_data
    assert filtered.additional_data["nested"] is not assembly.additional_data["nested"]

    filtered.additional_data["nested"]["values"].append(3)
    filtered.set_hidden_by_default("kept_datum", hidden=False)

    assert assembly.additional_data == {"nested": {"values": [1, 2]}}
    assert assembly.hidden_by_default_names == ["removed_datum", "kept_datum"]
    assert assembly.follower_indices_by_name == {
        "removed_follower": 1,
        "kept_follower": 3,
    }


def test_filtered_copy_with_no_exclusions_matches_copy():
    assembly = _assembly("datum")
    assembly.followers.append(create_box(1, 1, 1))
    assembly.add_named_direction_vector((1, 2, 3), "normal")

    copied = assembly.copy()
    filtered = assembly.filtered_copy(())

    assert filtered.follower_indices_by_name == copied.follower_indices_by_name
    assert filtered.cutter_indices_by_name == copied.cutter_indices_by_name
    assert (
        filtered.non_production_indices_by_name == copied.non_production_indices_by_name
    )
    assert (
        filtered.direction_vector_indices_by_name
        == copied.direction_vector_indices_by_name
    )
    assert filtered.direction_vectors == copied.direction_vectors
    assert filtered.hidden_by_default_names == copied.hidden_by_default_names
    assert filtered.additional_data == copied.additional_data
    assert filtered.leader is not assembly.leader
    assert filtered.non_production_parts[0] is not assembly.non_production_parts[0]


@pytest.mark.parametrize("excluded_names", [None, "datum", {"datum"}])
def test_filtered_copy_requires_list_or_tuple(excluded_names):
    with pytest.raises(TypeError, match="must be a list or tuple"):
        _assembly("datum").filtered_copy(excluded_names)


def test_filtered_copy_rejects_non_string_duplicate_and_unknown_names():
    assembly = _assembly("datum")

    with pytest.raises(TypeError, match="must contain strings"):
        assembly.filtered_copy(["datum", 42])
    with pytest.raises(ValueError, match="must not contain duplicates"):
        assembly.filtered_copy(["datum", "datum"])
    with pytest.raises(KeyError, match="Unknown artifact names.*missing"):
        assembly.filtered_copy(["missing"])
