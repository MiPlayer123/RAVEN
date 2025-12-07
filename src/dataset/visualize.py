# -*- coding: utf-8 -*-

"""
Visualization script for RAVEN dataset samples.

Usage:
    # Visualize a single NPZ file
    python visualize.py --input path/to/file.npz --output output.png

    # Visualize multiple NPZ files from a directory
    python visualize.py --input-dir path/to/dataset/center_single --output-dir ./visualizations

    # Visualize first N samples from each configuration
    python visualize.py --dataset-dir ./dataset --output-dir ./visualizations --num-samples 10
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def visualize_sample(npz_path, output_path=None, show=False):
    """
    Visualize a single RAVEN sample from an NPZ file.

    Args:
        npz_path: Path to the NPZ file
        output_path: Path to save the visualization (optional)
        show: Whether to display the image (optional)

    Returns:
        PIL Image object
    """
    data = np.load(npz_path)
    images = data['image']  # Shape: (16, 160, 160)
    target = int(data['target'])

    panel_size = 160
    gap = 5

    # Calculate dimensions
    context_width = 3 * panel_size + 2 * gap
    context_height = 3 * panel_size + 2 * gap
    answer_width = 8 * panel_size + 7 * gap
    total_width = max(context_width, answer_width)
    total_height = context_height + 40 + panel_size  # context + gap + answers

    # Create white canvas
    canvas = np.ones((total_height, total_width), dtype=np.uint8) * 255

    # Place context panels (first 8 images in 3x3 grid)
    for i in range(8):
        row = i // 3
        col = i % 3
        y = row * (panel_size + gap)
        x = col * (panel_size + gap)
        canvas[y:y+panel_size, x:x+panel_size] = images[i]

    # Draw border for the "?" cell (position 8 in the 3x3 grid)
    y = 2 * (panel_size + gap)
    x = 2 * (panel_size + gap)
    border_width = 2
    canvas[y:y+border_width, x:x+panel_size] = 0
    canvas[y+panel_size-border_width:y+panel_size, x:x+panel_size] = 0
    canvas[y:y+panel_size, x:x+border_width] = 0
    canvas[y:y+panel_size, x+panel_size-border_width:x+panel_size] = 0

    # Place answer choices (images 8-15)
    answer_y = context_height + 30
    for i in range(8):
        x = i * (panel_size + gap)
        canvas[answer_y:answer_y+panel_size, x:x+panel_size] = images[8 + i]

        # Highlight correct answer with a thicker border
        if i == target:
            border_width = 4
            canvas[answer_y:answer_y+border_width, x:x+panel_size] = 50
            canvas[answer_y+panel_size-border_width:answer_y+panel_size, x:x+panel_size] = 50
            canvas[answer_y:answer_y+panel_size, x:x+border_width] = 50
            canvas[answer_y:answer_y+panel_size, x+panel_size-border_width:x+panel_size] = 50

    # Convert to PIL Image
    img = Image.fromarray(canvas)

    # Add labels
    try:
        draw = ImageDraw.Draw(img)
        # Label for context
        draw.text((5, context_height + 5), "Answers (correct highlighted):", fill=0)
        # Label showing target
        draw.text((total_width - 150, context_height + 5), f"Target: {target}", fill=0)
    except Exception:
        pass  # Skip labels if font not available

    if output_path:
        img.save(output_path)
        print(f"Saved: {output_path}")

    if show:
        img.show()

    return img


def visualize_directory(input_dir, output_dir, num_samples=None):
    """
    Visualize NPZ files from a directory.

    Args:
        input_dir: Directory containing NPZ files
        output_dir: Directory to save visualizations
        num_samples: Maximum number of samples to visualize (None for all)
    """
    os.makedirs(output_dir, exist_ok=True)

    npz_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npz')])

    if num_samples:
        npz_files = npz_files[:num_samples]

    for npz_file in npz_files:
        input_path = os.path.join(input_dir, npz_file)
        output_path = os.path.join(output_dir, npz_file.replace('.npz', '.png'))
        visualize_sample(input_path, output_path)

    print(f"Visualized {len(npz_files)} samples from {input_dir}")


def visualize_dataset(dataset_dir, output_dir, num_samples=10):
    """
    Visualize samples from all configurations in a dataset.

    Args:
        dataset_dir: Root directory of the dataset
        output_dir: Directory to save visualizations
        num_samples: Number of samples per configuration
    """
    configs = [
        "center_single",
        "distribute_four",
        "distribute_nine",
        "left_center_single_right_center_single",
        "up_center_single_down_center_single",
        "in_center_single_out_center_single",
        "in_distribute_four_out_center_single"
    ]

    for config in configs:
        config_dir = os.path.join(dataset_dir, config)
        if os.path.exists(config_dir):
            config_output_dir = os.path.join(output_dir, config)
            visualize_directory(config_dir, config_output_dir, num_samples)
        else:
            print(f"Warning: Configuration directory not found: {config_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize RAVEN dataset samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Visualize a single file
    python visualize.py --input sample.npz --output sample.png

    # Visualize a single file and display it
    python visualize.py --input sample.npz --show

    # Visualize all files in a directory
    python visualize.py --input-dir ./dataset/center_single --output-dir ./viz

    # Visualize first 10 samples from each configuration
    python visualize.py --dataset-dir ./dataset --output-dir ./viz --num-samples 10
        """
    )

    parser.add_argument("--input", "-i", type=str,
                        help="Path to a single NPZ file")
    parser.add_argument("--output", "-o", type=str,
                        help="Output path for single file visualization")
    parser.add_argument("--show", "-s", action="store_true",
                        help="Display the visualization")
    parser.add_argument("--input-dir", type=str,
                        help="Directory containing NPZ files to visualize")
    parser.add_argument("--output-dir", type=str,
                        help="Directory to save visualizations")
    parser.add_argument("--dataset-dir", type=str,
                        help="Root dataset directory (visualize all configurations)")
    parser.add_argument("--num-samples", "-n", type=int, default=10,
                        help="Number of samples to visualize per configuration (default: 10)")

    args = parser.parse_args()

    # Validate arguments
    if args.input:
        # Single file mode
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            sys.exit(1)
        visualize_sample(args.input, args.output, args.show)

    elif args.input_dir:
        # Directory mode
        if not os.path.exists(args.input_dir):
            print(f"Error: Input directory not found: {args.input_dir}")
            sys.exit(1)
        if not args.output_dir:
            print("Error: --output-dir required when using --input-dir")
            sys.exit(1)
        visualize_directory(args.input_dir, args.output_dir, args.num_samples)

    elif args.dataset_dir:
        # Full dataset mode
        if not os.path.exists(args.dataset_dir):
            print(f"Error: Dataset directory not found: {args.dataset_dir}")
            sys.exit(1)
        if not args.output_dir:
            print("Error: --output-dir required when using --dataset-dir")
            sys.exit(1)
        visualize_dataset(args.dataset_dir, args.output_dir, args.num_samples)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
