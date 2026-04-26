import argparse
import struct
import numpy as np
from plyfile import PlyData


def parse_args():
    parser = argparse.ArgumentParser(description="Convert .ply to .splat")
    parser.add_argument("--ply", required=True, help="Input .ply file")
    parser.add_argument("--out", required=True, help="Output .splat file")
    return parser.parse_args()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sh_dc_to_rgb(sh):
    """Convert SH DC coefficient to [0, 255] uint8."""
    return np.clip((0.5 + 0.28209 * sh) * 255, 0, 255).astype(np.uint8)


def convert(input_path, output_path):
    print(f"Reading {input_path} …")
    plydata = PlyData.read(input_path)
    verts   = plydata["vertex"]
    n       = len(verts["x"])
    print(f"  {n:,} Gaussians found")

    # Position
    xs = np.array(verts["x"], dtype=np.float32)
    ys = np.array(verts["y"], dtype=np.float32)
    zs = np.array(verts["z"], dtype=np.float32)

    # Scale (stored as log in the PLY)
    s0 = np.exp(np.array(verts["scale_0"], dtype=np.float32))
    s1 = np.exp(np.array(verts["scale_1"], dtype=np.float32))
    s2 = np.exp(np.array(verts["scale_2"], dtype=np.float32))

    # Colour from SH DC term (f_dc_0/1/2) + opacity
    r = sh_dc_to_rgb(np.array(verts["f_dc_0"]))
    g = sh_dc_to_rgb(np.array(verts["f_dc_1"]))
    b = sh_dc_to_rgb(np.array(verts["f_dc_2"]))
    a = (sigmoid(np.array(verts["opacity"])) * 255).astype(np.uint8)

    # Rotation quaternion packed as uint8
    rot0 = np.array(verts["rot_0"], dtype=np.float32)
    rot1 = np.array(verts["rot_1"], dtype=np.float32)
    rot2 = np.array(verts["rot_2"], dtype=np.float32)
    rot3 = np.array(verts["rot_3"], dtype=np.float32)

    # Normalise quaternion
    norm = np.sqrt(rot0**2 + rot1**2 + rot2**2 + rot3**2) + 1e-8
    rot0 /= norm; rot1 /= norm; rot2 /= norm; rot3 /= norm

    pack_rot = lambda q: np.clip((q + 1) / 2 * 255, 0, 255).astype(np.uint8)
    pr0, pr1, pr2, pr3 = pack_rot(rot0), pack_rot(rot1), pack_rot(rot2), pack_rot(rot3)

    print(f"Writing {output_path} …")
    with open(output_path, "wb") as f:
        for i in range(n):
            # 3×float32 position + 3×float32 scale + 4×uint8 colour + 4×uint8 rotation
            f.write(struct.pack("<fff", xs[i], ys[i], zs[i]))
            f.write(struct.pack("<fff", s0[i], s1[i], s2[i]))
            f.write(bytes([r[i], g[i], b[i], a[i]]))
            f.write(bytes([pr0[i], pr1[i], pr2[i], pr3[i]]))

    size_mb = (n * (3*4 + 3*4 + 4 + 4)) / 1024**2
    print(f"Done. Output size ≈ {size_mb:.1f} MB  ({n:,} Gaussians × 32 bytes)")


def main():
    args = parse_args()
    convert(args.ply, args.out)


if __name__ == "__main__":
    main()
