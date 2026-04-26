import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plyfile import PlyData
from scipy import stats as scipy_stats
from sklearn.neighbors import NearestNeighbors



def parse_args():
    parser = argparse.ArgumentParser(description="Analyse Gaussian Splatting PLY file")
    parser.add_argument(
        "--ply",
        required=True,
        help="Path to point_cloud.ply (e.g. output/2a9c9eef-7/point_cloud/iteration_30000/point_cloud.ply)",
    )
    parser.add_argument(
        "--out_dir",
        default=".",
        help="Directory to save output images and CSV (default: current dir)",
    )
    return parser.parse_args()

def load_ply(path):
    plydata = PlyData.read(path)
    verts = plydata["vertex"]
    return verts

def compute_shape_metrics(verts):
    scale_0 = np.exp(np.array(verts["scale_0"]))
    scale_1 = np.exp(np.array(verts["scale_1"]))
    scale_2 = np.exp(np.array(verts["scale_2"]))

    print("Scale_0 stats:", scale_0.min(), scale_0.mean(), scale_0.max())
    print("Scale_1 stats:", scale_1.min(), scale_1.mean(), scale_1.max())
    print("Scale_2 stats:", scale_2.min(), scale_2.mean(), scale_2.max())

    max_scale = np.maximum(scale_0, np.maximum(scale_1, scale_2))
    min_scale = np.minimum(scale_0, np.minimum(scale_1, scale_2))
    mid_scale = scale_0 + scale_1 + scale_2 - max_scale - min_scale

    # Filter out degenerate Gaussians (near-zero scales)
    valid = min_scale > 1e-6
    print(f"Valid Gaussians: {valid.sum()} / {len(valid)} ({valid.mean()*100:.1f}%)")

    anisotropy_ratio = max_scale[valid] / min_scale[valid]
    elongation       = max_scale[valid] / mid_scale[valid]
    flatness         = mid_scale[valid] / min_scale[valid]

    print(f"\nAfter filtering:")
    print(f"Mean anisotropy ratio:   {anisotropy_ratio.mean():.3f}")
    print(f"Median anisotropy ratio: {np.median(anisotropy_ratio):.3f}")
    print(f"% ratio > 5:  {(anisotropy_ratio > 5).mean()*100:.1f}%")
    print(f"% ratio > 10: {(anisotropy_ratio > 10).mean()*100:.1f}%")

    return valid, anisotropy_ratio, elongation, flatness


def classify_shapes(anisotropy_ratio, elongation, flatness):
    needle = (elongation > 5) & (flatness < 2)
    disk   = (elongation < 2) & (flatness > 5)
    blob   = (elongation < 2) & (flatness < 2)
    other  = ~(needle | disk | blob)

    total = len(anisotropy_ratio)
    print(f"\nGaussian shape categories:")
    print(f"  Needle-like (elongated): {needle.sum()/total*100:.1f}%")
    print(f"  Disk-like   (flat):      {disk.sum()/total*100:.1f}%")
    print(f"  Blob-like   (isotropic): {blob.sum()/total*100:.1f}%")
    print(f"  Mixed:                   {other.sum()/total*100:.1f}%")

    return needle, disk, blob, other


def plot_anisotropy_distributions(anisotropy_ratio, elongation, flatness, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    cap = np.percentile(anisotropy_ratio, 99)
    axes[0].hist(anisotropy_ratio[anisotropy_ratio < cap], bins=50, log=True)
    axes[0].set_xlabel("Anisotropy Ratio (max/min)")
    axes[0].set_ylabel("Count (log)")
    axes[0].set_title("Anisotropy Ratio (capped at 99th pct)")

    axes[1].hist(elongation[elongation < np.percentile(elongation, 99)], bins=50, log=True)
    axes[1].set_xlabel("Elongation (max/mid)")
    axes[1].set_title("Elongation Distribution")

    axes[2].hist(flatness[flatness < np.percentile(flatness, 99)], bins=50, log=True)
    axes[2].set_xlabel("Flatness (mid/min)")
    axes[2].set_title("Flatness Distribution")

    plt.tight_layout()
    path = f"{out_dir}/anisotropy_fixed.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")


def plot_joint_distribution(elongation, flatness, out_dir):
    plt.figure(figsize=(7, 6))
    plt.hexbin(np.log1p(elongation), np.log1p(flatness),
               gridsize=50, cmap="YlOrRd", mincnt=1)
    plt.colorbar(label="Count (log)")
    plt.xlabel("log(1 + Elongation)")
    plt.ylabel("log(1 + Flatness)")
    plt.title("Joint Distribution of Elongation vs Flatness")
    plt.axvline(np.log1p(5), color="blue",  linestyle="--", label="Elongation=5")
    plt.axhline(np.log1p(5), color="green", linestyle="--", label="Flatness=5")
    plt.legend()
    path = f"{out_dir}/joint_anisotropy.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")


def plot_spatial_categories(x, y, z, needle, disk, blob, other, out_dir):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5),
                              subplot_kw={"projection": "3d"})

    categories = {
        "Needle": (needle, "red"),
        "Disk":   (disk,   "blue"),
        "Blob":   (blob,   "green"),
        "Mixed":  (other,  "orange"),
    }

    for ax, (label, (mask, color)) in zip(axes, categories.items()):
        ax.scatter(x[mask], y[mask], z[mask], c=color, s=0.1, alpha=0.3)
        ax.set_title(label)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    plt.suptitle("Spatial Distribution by Gaussian Shape Category", y=1.02)
    plt.tight_layout()
    path = f"{out_dir}/spatial_categories.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")


def plot_3d_anisotropy(x, y, z, anisotropy_ratio, out_dir):
    cap_ratio = np.clip(anisotropy_ratio, 1, np.percentile(anisotropy_ratio, 95))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, c=cap_ratio, cmap="hot", s=0.5)
    plt.colorbar(sc, label="Anisotropy Ratio (capped)")
    ax.set_title("Spatial Distribution of Anisotropy")
    path = f"{out_dir}/anisotropy_3d_fixed.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")


def plot_spatial_analysis(x, y, z, anisotropy_ratio, high_aniso_mask, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(y, bins=100, color="steelblue", alpha=0.7, label="All")
    axes[0].hist(y[high_aniso_mask], bins=100, color="red", alpha=0.7, label="High aniso (top 10%)")
    axes[0].set_xlabel("Y coordinate (vertical)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Vertical Distribution of Gaussians")
    axes[0].legend()

    axes[1].hexbin(y, np.log1p(anisotropy_ratio), gridsize=50, cmap="YlOrRd", mincnt=1)
    axes[1].set_xlabel("Y coordinate (vertical)")
    axes[1].set_ylabel("log(1 + Anisotropy Ratio)")
    axes[1].set_title("Anisotropy vs. Height")

    sc = axes[2].scatter(x, z, c=np.log1p(anisotropy_ratio), cmap="hot", s=0.3, alpha=0.5)
    plt.colorbar(sc, ax=axes[2], label="log(1+Anisotropy)")
    axes[2].set_xlabel("X"); axes[2].set_ylabel("Z")
    axes[2].set_title("Top-Down View (Anisotropy)")

    plt.tight_layout()
    path = f"{out_dir}/spatial_analysis.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")


# ── Statistical fitting ────────────────────────────────────────────────────

def fit_lognormal(anisotropy_ratio, out_dir):
    log_ratios = np.log(anisotropy_ratio[anisotropy_ratio > 1])
    mu, std = scipy_stats.norm.fit(log_ratios)

    print(f"Log-normal fit: μ = {mu:.3f}, σ = {std:.3f}")
    print(f"Implied median ratio: {np.exp(mu):.2f}")
    print(f"Implied mean ratio:   {np.exp(mu + std**2/2):.2f}")

    stat, p = scipy_stats.kstest(log_ratios, scipy_stats.norm(mu, std).cdf)
    print(f"\nKS test: statistic={stat:.4f}, p={p:.4f}")
    print("Log-normal fit:", "rejected" if p < 0.05 else "not rejected", "at 5% level")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(log_ratios, bins=100, density=True, alpha=0.6, label="Data")
    x_range = np.linspace(log_ratios.min(), log_ratios.max(), 200)
    ax.plot(x_range, scipy_stats.norm.pdf(x_range, mu, std),
            "r-", lw=2, label=f"Log-Normal(μ={mu:.2f}, σ={std:.2f})")
    ax.set_xlabel("log(Anisotropy Ratio)")
    ax.set_ylabel("Density")
    ax.set_title("Log-Normal Fit to Anisotropy Ratio Distribution")
    ax.legend()
    path = f"{out_dir}/lognormal_fit.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")



def spatial_clustering_analysis(coords, anisotropy_ratio):
    high_aniso_mask = anisotropy_ratio > np.percentile(anisotropy_ratio, 90)
    low_aniso_mask  = ~high_aniso_mask

    high_coords = coords[high_aniso_mask]
    low_coords  = coords[low_aniso_mask]

    def mean_nn_distance(pts, k=5):
        nn = NearestNeighbors(n_neighbors=k + 1).fit(pts)
        dists, _ = nn.kneighbors(pts)
        return dists[:, 1:].mean()

    np.random.seed(42)
    sample_size = min(5000, len(high_coords), len(low_coords))
    hi_sample = high_coords[np.random.choice(len(high_coords), sample_size, replace=False)]
    lo_sample = low_coords [np.random.choice(len(low_coords),  sample_size, replace=False)]

    hi_nn = mean_nn_distance(hi_sample)
    lo_nn = mean_nn_distance(lo_sample)

    print(f"Mean 5-NN distance (high anisotropy): {hi_nn:.4f}")
    print(f"Mean 5-NN distance (low anisotropy):  {lo_nn:.4f}")
    print(f"Ratio: {lo_nn/hi_nn:.2f}x  →  high-aniso points are "
          f"{'more' if hi_nn < lo_nn else 'less'} clustered")

    return high_aniso_mask


def local_density_analysis(coords, anisotropy_ratio, out_dir):
    nn = NearestNeighbors(n_neighbors=10).fit(coords)
    dists, indices = nn.kneighbors(coords)

    local_density   = 1.0 / (dists[:, 1:].mean(axis=1) + 1e-8)
    local_aniso_std = np.array([
        anisotropy_ratio[indices[i, 1:]].std()
        for i in range(len(coords))
    ])

    print("Correlation between local density and anisotropy:")
    print(f"  Pearson r = {np.corrcoef(local_density, anisotropy_ratio)[0,1]:.4f}")
    print("\nCorrelation between local aniso variation and anisotropy:")
    print(f"  Pearson r = {np.corrcoef(local_aniso_std, anisotropy_ratio)[0,1]:.4f}")

    plt.figure(figsize=(7, 5))
    plt.hexbin(np.log1p(local_density), np.log1p(anisotropy_ratio),
               gridsize=50, cmap="YlOrRd", mincnt=1)
    plt.colorbar(label="Count")
    plt.xlabel("log(1 + Local Density)")
    plt.ylabel("log(1 + Anisotropy Ratio)")
    plt.title("Anisotropy vs. Local Point Density")
    path = f"{out_dir}/density_vs_anisotropy.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")


def save_summary(anisotropy_ratio, elongation, flatness, out_dir):
    def stats(arr):
        return {
            "Count":  len(arr),
            "Mean":   arr.mean(),
            "Median": np.median(arr),
            "Std":    arr.std(),
            "P25":    np.percentile(arr, 25),
            "P75":    np.percentile(arr, 75),
            "P99":    np.percentile(arr, 99),
        }

    summary = pd.DataFrame({
        "Anisotropy Ratio": stats(anisotropy_ratio),
        "Elongation":       stats(elongation),
        "Flatness":         stats(flatness),
    }).T

    print(summary.to_string())
    path = f"{out_dir}/anisotropy_summary.csv"
    summary.to_csv(path)
    print(f"Saved: {path}")


def main():
    args = parse_args()

    print(f"Loading PLY: {args.ply}")
    verts = load_ply(args.ply)

    valid, anisotropy_ratio, elongation, flatness = compute_shape_metrics(verts)

    x = np.array(verts["x"])[valid]
    y = np.array(verts["y"])[valid]
    z = np.array(verts["z"])[valid]
    coords = np.stack([x, y, z], axis=1)

    needle, disk, blob, other = classify_shapes(anisotropy_ratio, elongation, flatness)

    plot_anisotropy_distributions(anisotropy_ratio, elongation, flatness, args.out_dir)
    plot_joint_distribution(elongation, flatness, args.out_dir)
    plot_spatial_categories(x, y, z, needle, disk, blob, other, args.out_dir)
    plot_3d_anisotropy(x, y, z, anisotropy_ratio, args.out_dir)

    fit_lognormal(anisotropy_ratio, args.out_dir)

    high_aniso_mask = spatial_clustering_analysis(coords, anisotropy_ratio)
    plot_spatial_analysis(x, y, z, anisotropy_ratio, high_aniso_mask, args.out_dir)
    local_density_analysis(coords, anisotropy_ratio, args.out_dir)

    save_summary(anisotropy_ratio, elongation, flatness, args.out_dir)


if __name__ == "__main__":
    main()
