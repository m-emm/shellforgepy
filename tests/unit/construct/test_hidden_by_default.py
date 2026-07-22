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
