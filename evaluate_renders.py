import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from scipy.ndimage import uniform_filter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Gaussian Splatting renders")
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to the trained model directory (e.g. output/2a9c9eef-7)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split to evaluate: train or test (default: train)",
    )
    parser.add_argument(
        "--iteration",
        default="30000",
        help="Iteration number used in the render output path (default: 30000)",
    )
    parser.add_argument(
        "--variance_percentile",
        type=float,
        default=70.0,
        help="Percentile threshold for classifying high-variance (anisotropic) pixels (default: 70)",
    )
    parser.add_argument(
        "--out_dir",
        default=".",
        help="Directory to save output images and CSV (default: current dir)",
    )
    return parser.parse_args()


def load_frame(path):
    return np.array(Image.open(path)) / 255.0


def local_variance_mask(gt_img, window=15, percentile=70.0):
    """Return a boolean mask of pixels above the given variance percentile."""
    gray  = gt_img.mean(axis=2)
    mean  = uniform_filter(gray, window)
    mean2 = uniform_filter(gray ** 2, window)
    var_map = mean2 - mean ** 2
    return var_map, var_map > np.percentile(var_map, percentile)


def evaluate_all_frames(render_dir, gt_dir, variance_percentile):
    frames = sorted(os.listdir(render_dir))
    results = []

    for frame in frames:
        rendered = load_frame(os.path.join(render_dir, frame))
        gt       = load_frame(os.path.join(gt_dir,     frame))

        var_map, mask = local_variance_mask(gt, percentile=variance_percentile)

        psnr_aniso = peak_signal_noise_ratio(gt[mask],  rendered[mask],  data_range=1.0)
        psnr_iso   = peak_signal_noise_ratio(gt[~mask], rendered[~mask], data_range=1.0)
        ssim_full  = structural_similarity(gt, rendered, channel_axis=2, data_range=1.0)

        results.append({
            "frame":      frame,
            "psnr_aniso": psnr_aniso,
            "psnr_iso":   psnr_iso,
            "psnr_diff":  psnr_iso - psnr_aniso,
            "ssim":       ssim_full,
            "aniso_frac": mask.mean(),
        })

    return pd.DataFrame(results), frames


def visualise_representative_frame(render_dir, gt_dir, frames, df,
                                   variance_percentile, out_dir):
    frame    = frames[len(frames) // 2]
    rendered = load_frame(os.path.join(render_dir, frame))
    gt       = load_frame(os.path.join(gt_dir,     frame))

    var_map, mask = local_variance_mask(gt, percentile=variance_percentile)
    diff = np.abs(gt - rendered).mean(axis=2)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    axes[0, 0].imshow(gt);              axes[0, 0].set_title("Ground Truth")
    axes[0, 1].imshow(rendered);        axes[0, 1].set_title("Rendered")
    axes[0, 2].imshow(mask, cmap="gray")
    axes[0, 2].set_title(f"Anisotropic Mask\n(top {100-variance_percentile:.0f}% variance)")

    im = axes[1, 0].imshow(diff, cmap="hot", vmin=0, vmax=0.1)
    plt.colorbar(im, ax=axes[1, 0])
    axes[1, 0].set_title("Absolute Error Map")

    axes[1, 1].imshow(diff, cmap="hot", vmin=0, vmax=0.1)
    axes[1, 1].contour(mask, colors="cyan", linewidths=0.8)
    axes[1, 1].set_title("Error + Anisotropic Boundary")

    means = [df["psnr_aniso"].mean(), df["psnr_iso"].mean()]
    axes[1, 2].bar(["Anisotropic\nRegions", "Isotropic\nRegions"],
                   means, color=["coral", "steelblue"], width=0.5)
    axes[1, 2].set_ylabel("Mean PSNR (dB)")
    axes[1, 2].set_title("Reconstruction Quality\nby Region Type")
    axes[1, 2].set_ylim(0, max(means) * 1.2)
    for i, val in enumerate(means):
        axes[1, 2].text(i, val + 0.3, f"{val:.2f} dB", ha="center", fontweight="bold")

    for ax in axes.flat:
        if ax != axes[1, 2]:
            ax.axis("off")

    plt.tight_layout()
    path = f"{out_dir}/psnr_analysis.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")


def visualise_variance_mask(render_dir, gt_dir, frames, variance_percentile, out_dir):
    frame = frames[0]
    gt    = load_frame(os.path.join(gt_dir, frame))
    var_map, mask = local_variance_mask(gt, percentile=variance_percentile)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(gt);                        axes[0].set_title("Ground Truth")
    im = axes[1].imshow(var_map, cmap="hot");  plt.colorbar(im, ax=axes[1])
    axes[1].set_title("Local Variance Map")
    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title(f"Anisotropic Mask\n(top {100-variance_percentile:.0f}% variance)")
    plt.tight_layout()
    path = f"{out_dir}/variance_mask.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")


def main():
    args = parse_args()

    render_dir = os.path.join(
        args.model_dir, args.split, f"ours_{args.iteration}", "renders"
    )
    gt_dir = os.path.join(
        args.model_dir, args.split, f"ours_{args.iteration}", "gt"
    )

    for d in (render_dir, gt_dir):
        if not os.path.isdir(d):
            raise FileNotFoundError(
                f"Directory not found: {d}\n"
                f"Make sure you have run: python gaussian-splatting/render.py -m {args.model_dir}"
            )

    print(f"Render dir : {render_dir}")
    print(f"GT dir     : {gt_dir}")

    df, frames = evaluate_all_frames(render_dir, gt_dir, args.variance_percentile)

    print("\n── Per-frame results ──")
    print(df.to_string(index=False))
    print("\n── Aggregate ──")
    print(df.describe().round(3))

    print("\n" + "=" * 50)
    print(f"Mean PSNR  (anisotropic regions): {df['psnr_aniso'].mean():.2f} dB")
    print(f"Mean PSNR  (isotropic regions):   {df['psnr_iso'].mean():.2f} dB")
    print(f"Mean gap   (iso - aniso):         {df['psnr_diff'].mean():.2f} dB")
    print(f"Mean SSIM  (full image):          {df['ssim'].mean():.4f}")
    print("=" * 50)

    csv_path = f"{args.out_dir}/psnr_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results: {csv_path}")

    visualise_variance_mask(render_dir, gt_dir, frames, args.variance_percentile, args.out_dir)
    visualise_representative_frame(render_dir, gt_dir, frames, df,
                                   args.variance_percentile, args.out_dir)


if __name__ == "__main__":
    main()
