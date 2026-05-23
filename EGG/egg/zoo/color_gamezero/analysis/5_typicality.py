
#=========================a function that computes dist2name and dist2other for each color patch ======================


import pickle
import numpy as np
import colorspacious as cs
import csv
from plot_utils import extract_color_data, compute_cielab_distance, load_prototypes, hls_to_rgb, extract_color_data_all_label
import argparse


def compute_distances(data, prototypes):
    """
    Compute:
    - dist2name: distance from target color to its assigned prototype.
    - dist2other: distance from target color to the nearest alternative prototype.
    - dist2others: average distance from target color to all other prototypes.
    
    Returns:
    - A list of results containing (color_name, LAB, dist2name, dist2other, dist2others).
    - The average dist2name, dist2other, and dist2others across all data points.
    """
    results = []
    total_dist2name = 0
    total_dist2other = 0
    total_dist2others = 0
    count = 0

    for hls, color_name in data:
        rgb = hls_to_rgb(*hls)
        target_lab = cs.cspace_convert(rgb, start="sRGB255", end="CIELab")  # Convert to CIELAB
        prototype_lab = prototypes.get(color_name, None)  # Get the prototype for this name

        if prototype_lab is None:
            continue  # Skip if no prototype exists

        # Compute dist2name
        dist2name = compute_cielab_distance(target_lab, prototype_lab)

        # Compute dist2other (nearest alternative prototype)
        other_prototypes = {name: lab for name, lab in prototypes.items() if name != color_name}
        if other_prototypes:
            dist2other = min(compute_cielab_distance(target_lab, other_lab) for other_lab in other_prototypes.values())

            # Compute dist2others (average distance to all other prototypes)
            dist2others = np.mean([compute_cielab_distance(target_lab, other_lab) for other_lab in other_prototypes.values()])
        else:
            dist2other = None  # No other prototypes to compare
            dist2others = None  # No other prototypes to compare

        results.append((color_name, target_lab, dist2name, dist2other, dist2others))

        # Accumulate for average calculation
        total_dist2name += dist2name
        if dist2other is not None:
            total_dist2other += dist2other
        if dist2others is not None:
            total_dist2others += dist2others
        count += 1

    # Compute averages
    avg_dist2name = total_dist2name / count if count > 0 else None
    avg_dist2other = total_dist2other / count if count > 0 else None
    avg_dist2others = total_dist2others / count if count > 0 else None

    return results, avg_dist2name, avg_dist2other, avg_dist2others


def save_distance_results(results, avg_dist2name, avg_dist2other, avg_dist2others, output_csv):
    """Save distance calculations to a CSV file for inspection and print averages."""
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["color_name", "L", "A", "B", "dist2name", "dist2other", "avg_dist2others"])  # Header
        for name, lab, d2n, d2o, d2os in results:
            writer.writerow([name] + list(lab) + [d2n, d2o, d2os])
    
    print(f"Results saved successfully to {output_csv}!")
    print(f"Average dist2name: {avg_dist2name:.3f}")
    print(f"Average dist2other: {avg_dist2other:.3f}")
    print(f"Average dist2others: {avg_dist2others:.3f}")


def analyze_typicality(data_file, prototypes_file, output_csv, rl, use_label):
    """
    Compute typicality measures (dist2name & dist2other), save the results, and print averages.
    """
    prototypes = load_prototypes(prototypes_file)
    if data_file.endswith(".csv"):
        color_data_by_condition, _, all_color_data, _ = extract_color_data_all_label(data_file)
    else:
        color_data_by_condition, _, all_color_data, _ = extract_color_data(data_file, rl, use_label)

    # Define all data splits to process
    data_splits = {
        "all": all_color_data,
        "far": color_data_by_condition["far"],
        "split": color_data_by_condition["split"],
        "close": color_data_by_condition["close"]
    }

    # Loop through and compute/save distances
    for condition, data in data_splits.items():
        results, avg_dist2name, avg_dist2other, avg_dist2others = compute_distances(data, prototypes)
        save_distance_results(results, avg_dist2name, avg_dist2other, avg_dist2others, f"{output_csv}_{condition}.csv")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze typicality of color names.")
    parser.add_argument("--mode", choices=["model", "human"], required=True,
                        help="Specify whether the analysis is for model or human data.")
    parser.add_argument("--data_file", required=True, help="Path to the input data file.")
    parser.add_argument("--prototypes_file", required=True, help="Path to the prototypes .pkl file.")
    parser.add_argument("--output_csv", required=True, help="Path to save the output CSV.")
    parser.add_argument(
        "--rl_or_sl",
        action="store_true",
        help="Flag for rl or sl)."
    )
    parser.add_argument(
        "--use_label",
        action="store_true",
        help="Flag to use label (match.group(2)) instead of agent's word (match.group(1))."
    )

    args = parser.parse_args()

    analyze_typicality(args.data_file, args.prototypes_file, args.output_csv, args.rl_or_sl, args.use_label)
    # if args.mode == "human":
    #     analyze_typicality_human(args.data_file, args.prototypes_file, args.output_csv)

