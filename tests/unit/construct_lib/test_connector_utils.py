import numpy as np
from shellforgepy.shells.connector_hint import ConnectorHint
from shellforgepy.shells.connector_utils import compute_connector_hints_from_shell_maps


def test_connector_hints_from_manual_tetrahedron_shell_map():
    # Tetrahedron points
    V = {
        0: np.array([1.0, 1.0, 1.0]),
        1: np.array([-1.0, -1.0, 1.0]),
        2: np.array([-1.0, 1.0, -1.0]),
        3: np.array([1.0, -1.0, -1.0]),
    }

    # Tetrahedron faces
    mesh_faces = np.array(
        [
            [0, 1, 2],  # Face 0
            [0, 2, 3],  # Face 1
            [0, 3, 1],  # Face 2
            [1, 3, 2],  # Face 3
        ]
    )

    face_to_region = {
        0: 1,
        1: 2,
        2: 1,
        3: 2,
    }

    # Shrink center
    center = sum(V.values()) / 4
    shrink = lambda v: v - 0.1 * (v - center)

    shell_maps = {}
    vertex_index_map = {}

    for fid, face in enumerate(mesh_faces):
        local_verts = [shrink(V[i]) for i in face]
        shell_maps[fid] = {
            "vertexes": {i: v for i, v in enumerate(local_verts)},
            "faces": {0: [0, 1, 2]},
        }
        vertex_index_map[fid] = {"inner": {face[i]: i for i in range(3)}}

    hints = compute_connector_hints_from_shell_maps(
        mesh_faces=mesh_faces,
        face_to_region=face_to_region,
        shell_maps=shell_maps,
        vertex_index_map=vertex_index_map,
    )

    assert isinstance(hints, list)
    assert all(isinstance(h, ConnectorHint) for h in hints)
    assert all(h.region_a < h.region_b for h in hints)
    assert len(hints) > 0
