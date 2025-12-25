import pytest

pytest.importorskip("cadquery")

from shellforgepy.adapters.cadquery import cadquery_adapter
from shellforgepy.construct.leader_followers_cutters_part import (
    LeaderFollowersCuttersPart,
)
from shellforgepy.construct.step_serialization import (
    deserialize_to_leader_followers_cutters_part,
    serialize_to_step,
)
from shellforgepy.simple import *


def test_serialize_to_step(tmp_path):
    """Test serialization of LeaderFollowersCuttersPart to STEP file."""

    leader = create_box(10, 10, 10)
    follower = create_cylinder(5, 15)

    lfcp = LeaderFollowersCuttersPart(leader=leader, followers=[follower])
    step_file_path = tmp_path / "test_part.step"

    serialize_to_step(lfcp, str(step_file_path))

    assert step_file_path.is_file()
    assert step_file_path.stat().st_size > 0


def test_serialize_deserialize_round_trip_with_names(tmp_path):
    leader = create_box(10, 10, 10)
    follower_a = create_cylinder(2, 8)
    follower_b = create_cylinder(3, 6)
    cutter = create_cylinder(1, 12)
    non_prod = create_box(4, 4, 2)

    lfcp = LeaderFollowersCuttersPart(
        leader=leader,
        followers=[follower_a, follower_b],
        cutters=[cutter],
        non_production_parts=[non_prod],
        follower_names=["follower_a", "follower_b"],
        cutter_names=["cutter_a"],
        non_production_names=["non_prod_a"],
    )

    step_file_path = tmp_path / "round_trip.step"
    serialize_to_step(lfcp, str(step_file_path))

    restored = deserialize_to_leader_followers_cutters_part(str(step_file_path))

    assert restored.leader is not None
    if restored.followers or restored.cutters or restored.non_production_parts:
        assert len(restored.followers) == 2
        assert len(restored.cutters) == 1
        assert len(restored.non_production_parts) == 1

        assert restored.get_follower_index_by_name("follower_a") == 0
        assert restored.get_follower_index_by_name("follower_b") == 1
        assert restored.get_cutter_index_by_name("cutter_a") == 0
        assert restored.get_non_production_index_by_name("non_prod_a") == 0

        assert get_volume(restored.leader) == pytest.approx(10 * 10 * 10, rel=1e-4)
    else:
        solids = cadquery_adapter.extract_solids(restored.leader)
        assert len(solids) == 5
        volumes = sorted([solid.Volume() for solid in solids])
        expected = sorted(
            [
                10 * 10 * 10,
                4 * 4 * 2,
                32 * 3.141592653589793,
                54 * 3.141592653589793,
                12 * 3.141592653589793,
            ]
        )
        for vol, exp in zip(volumes, expected):
            assert vol == pytest.approx(exp, rel=1e-4)
