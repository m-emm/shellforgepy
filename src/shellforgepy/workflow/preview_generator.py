#!/usr/bin/env python3
# pure_numpy_stl_rasterizer.py
import struct

import numpy as np

try:
    from PIL import Image

    HAVE_PIL = True
except Exception:
    HAVE_PIL = False


# ---------- STL loader (binary) ----------
def load_binary_stl(path: str):
    with open(path, "rb") as f:
        header = f.read(80)
        (n_tri,) = struct.unpack("<I", f.read(4))
        # Each triangle: normal(3f) + v0(3f) + v1(3f) + v2(3f) + attr(2B)
        rec = np.fromfile(
            f,
            dtype=np.dtype(
                [
                    ("n", "<f4", (3,)),
                    ("v0", "<f4", (3,)),
                    ("v1", "<f4", (3,)),
                    ("v2", "<f4", (3,)),
                    ("attr", "<u2"),
                ]
            ),
            count=n_tri,
        )
    V = np.stack([rec["v0"], rec["v1"], rec["v2"]], axis=1)  # (N,3,3)
    N_face = rec["n"].astype(np.float32)  # (N,3)
    return V.astype(np.float32), N_face


# ---------- Camera & transforms ----------
def ortho_matrix(left, right, bottom, top, near, far):
    # Standard OpenGL-style orthographic projection
    M = np.eye(4, dtype=np.float32)
    M[0, 0] = 2.0 / (right - left)
    M[1, 1] = 2.0 / (top - bottom)
    M[2, 2] = -2.0 / (far - near)
    M[0, 3] = -(right + left) / (right - left)
    M[1, 3] = -(top + bottom) / (top - bottom)
    M[2, 3] = -(far + near) / (far - near)
    return M


def look_at(eye, target, up=(0, 0, 1)):
    eye = np.array(eye, np.float32)
    target = np.array(target, np.float32)
    up = np.array(up, np.float32)
    z = eye - target
    z /= np.linalg.norm(z) + 1e-12
    x = np.cross(up, z)
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(z, x)
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = np.stack([x, y, z], axis=0)
    M[:3, 3] = -M[:3, :3] @ eye
    return M


def mvp_transform(V, M):
    # V: (N,3,3), add w=1, apply M (4x4), perspective divide
    N = V.shape[0]
    V4 = np.concatenate([V, np.ones((N, 3, 1), np.float32)], axis=2)  # (N,3,4)
    Vt = (M @ V4.transpose(0, 2, 1)).transpose(0, 2, 1)  # (N,3,4)
    # For ortho, w==1, but keep generality:
    V_ndc = Vt[..., :3] / np.clip(Vt[..., 3:4], 1e-12, None)
    return V_ndc  # in [-1,1] cube


def ndc_to_screen(V_ndc, width, height):
    x = (V_ndc[..., 0] * 0.5 + 0.5) * (width - 1)
    y = (1.0 - (V_ndc[..., 1] * 0.5 + 0.5)) * (height - 1)  # flip Y
    z = V_ndc[..., 2]
    return np.stack([x, y, z], axis=-1)  # (N,3,3)


# ---------- Rasterize triangles with z-buffer ----------
def rasterize_triangles(
    screen_tris, face_normals, img_w, img_h, light_dir=(0.5, 0.7, -1.0)
):
    # Buffers
    depth = np.full((img_h, img_w), np.inf, dtype=np.float32)
    rgb = np.zeros((img_h, img_w, 3), dtype=np.float32)

    L = np.array(light_dir, np.float32)
    L /= np.linalg.norm(L) + 1e-12

    # Precompute per-face Lambert term in view space (use face normals)
    # Clamp to [0,1], add small ambient
    lambert = np.maximum(0.0, (face_normals @ -L)) * 0.9 + 0.1  # (N,)

    # Iterate triangles (vectorized per triangle bounding box)
    for i in range(screen_tris.shape[0]):
        tri = screen_tris[i]  # (3,3) -> (x,y,z)
        # Bounding box (clipped to screen)
        x0 = int(np.floor(np.clip(np.min(tri[:, 0]), 0, img_w - 1)))
        x1 = int(np.ceil(np.clip(np.max(tri[:, 0]), 0, img_w - 1)))
        y0 = int(np.floor(np.clip(np.min(tri[:, 1]), 0, img_h - 1)))
        y1 = int(np.ceil(np.clip(np.max(tri[:, 1]), 0, img_h - 1)))
        if x1 < x0 or y1 < y0:
            continue

        # Triangle setup
        # Edge function coefficients
        x = tri[:, 0]
        y = tri[:, 1]
        z = tri[:, 2]
        # Barycentric via area/edge functions
        denom = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
        if abs(denom) < 1e-12:
            continue

        xs = np.arange(x0, x1 + 1, dtype=np.float32)
        ys = np.arange(y0, y1 + 1, dtype=np.float32)
        XX, YY = np.meshgrid(xs, ys)  # (H_box, W_box)

        w0 = ((y[1] - y[2]) * (XX - x[2]) + (x[2] - x[1]) * (YY - y[2])) / denom
        w1 = ((y[2] - y[0]) * (XX - x[2]) + (x[0] - x[2]) * (YY - y[2])) / denom
        w2 = 1.0 - w0 - w1

        # Inside test: allow on-edge
        mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not np.any(mask):
            continue

        # Interpolate depth
        z_box = w0 * z[0] + w1 * z[1] + w2 * z[2]

        # Depth test
        sub_depth = depth[y0 : y1 + 1, x0 : x1 + 1]
        sub_rgb = rgb[y0 : y1 + 1, x0 : x1 + 1]
        closer = (z_box < sub_depth) & mask

        if np.any(closer):
            # Simple grayscale based on lambert term, tint optional
            shade = lambert[i]
            color = np.array([0.85, 0.88, 0.92], np.float32) * shade  # cool grey
            # Assign
            sub_depth[closer] = z_box[closer]
            sub_rgb[closer] = color

    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


# ---------- Putting it together ----------
def render_stl_to_png(
    stl_path,
    out_path="out.png",
    img_w=1024,
    img_h=1024,
    bed_mm=(220.0, 220.0, 220.0),
    model_offset=(0, 0, 0),
):
    V, N_face = load_binary_stl(stl_path)  # V: (N,3,3), world mm

    # Optional: recalc face normals if STL normals are junk
    def face_normals(V):
        e1 = V[:, 1] - V[:, 0]
        e2 = V[:, 2] - V[:, 0]
        n = np.cross(e1, e2)
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        return n.astype(np.float32)

    # Use STL normals if they look normalized, else recompute:
    if not np.allclose(np.linalg.norm(N_face, axis=1).mean(), 1.0, atol=1e-2):
        N_face = face_normals(V)

    model_bbox = np.array([V.min(axis=(0, 1)), V.max(axis=(0, 1))], np.float32)  # (2,3)
    model_center = 0.5 * (model_bbox[0] + model_bbox[1])

    model_offset = (
        np.array(model_offset, np.float32) + model_center + np.array([0, 0, 0])
    )  # put base at z=0

    # Center/offset model
    V = V + np.array(model_offset, np.float32)

    # Camera: orthographic top-down (z axis up), eye above center
    w, h, d = bed_mm
    center = np.array([w * 0.5, h * 0.5, 0.0], np.float32)
    eye = center + np.array([0, -250, 0.0], np.float32)  # any positive z
    view = look_at(eye, center, up=(0, 0, 1))
    proj = ortho_matrix(0, w, 0, h, near=-d, far=+d)  # fit bed area in mm
    MVP = proj @ view

    V_ndc = mvp_transform(V, MVP)  # (N,3,3) in NDC
    screen_tris = ndc_to_screen(V_ndc, img_w, img_h)  # (N,3,3)

    # Transform face normals into view space for lighting
    # For orthographic view with look_at, a 3x3 rotation of 'view' is sufficient.
    R = view[:3, :3]
    N_view = (R @ N_face.T).T
    N_view /= np.linalg.norm(N_view, axis=1, keepdims=True) + 1e-12

    img = rasterize_triangles(screen_tris, N_view, img_w, img_h)

    if HAVE_PIL:
        Image.fromarray(img).save(out_path)
    return img


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pure_numpy_stl_rasterizer.py model.stl [out.png]")
        sys.exit(1)
    stl_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "out.png"
    render_stl_to_png(stl_path, out_path)
    print(f"Wrote {out_path}")
