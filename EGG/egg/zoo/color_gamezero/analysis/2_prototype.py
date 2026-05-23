
import pickle
import numpy as np
import colorspacious as cs
import csv
from collections import defaultdict
import argparse
from plot_utils import extract_color_data, rgb_to_cielab, hls_to_rgb, cielab_to_rgb, load_prototypes
import os 


def compute_prototypes(data, file_path, pickle_output, csv_output):

    """Compute the prototype (mean CIELAB value) for each color name."""
    color_dict = defaultdict(list)

    # # Convert RGB to CIELAB and group by color name
    # for hls, color_name in data:
    #     rgb = hls_to_rgb(*hls)
    #     color_dict[color_name].append(rgb_to_cielab(*rgb))

   # Convert RGB to CIELAB and group by color name
    for cielab, color_name in data:
        color_dict[color_name].append(cielab)

    # Compute the mean prototype for each color name
    prototypes = {color: np.mean(np.array(lab_values), axis=0) for color, lab_values in color_dict.items()}

    # 🔹 Extract directory path from data_path
    save_path = os.path.dirname(os.path.abspath(file_path))
    print(f"Saving prototypes to directory: {save_path}")

    pickle_output_path = os.path.join(save_path, pickle_output)
    csv_output_path = os.path.join(save_path, csv_output)

    save_prototypes_pickle(prototypes, pickle_output_path)
    save_prototypes_csv(prototypes, csv_output_path)
    
    return prototypes

def save_prototypes_pickle(prototypes, output_path):
    """Save the prototypes dictionary as a pickle file."""
    with open(output_path, "wb") as f:
        pickle.dump(prototypes, f)
    print(f"Prototypes saved successfully to {output_path}!")

def save_prototypes_csv(prototypes, output_path):
    """Save the prototypes as a CSV file."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["color_name", "L", "A", "B"])  # Header row
        for color, lab_values in prototypes.items():
            writer.writerow([color] + list(lab_values))  # Write color name and L*a*b* values
    print(f"Prototypes saved successfully to {output_path}!")



#=========================================visualize prototype============================================


# def plot_color_prototypes(prototypes, pickle_path):
#     """Visualize color prototypes in a grid."""
#     fig, ax = plt.subplots(figsize=(50, 4))
    
#     # Sort colors alphabetically for consistency
#     color_names = sorted(prototypes.keys())

#     for i, color in enumerate(color_names):
#         lab_value = prototypes[color]  # Get CIELAB values
#         rgb_value = cielab_to_rgb(lab_value)  # Convert to RGB

#         ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=rgb_value))  # Draw color block
#         ax.text(i + 0.5, -0.3, color, ha='center', va='center', fontsize=14)  # Label

#     ax.set_xlim(0, len(color_names))
#     ax.set_ylim(0, 1)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_frame_on(False)  # Remove borders
#     output_file = f"{pickle_path}.png"

#     plt.savefig(output_file, bbox_inches='tight', dpi=300)


# def visualize_prototypes(pickle_path):
#     """Load prototypes from a file and visualize them."""
#     prototypes = load_prototypes(pickle_path)
#     plot_color_prototypes(prototypes, pickle_path)


def main():
    parser = argparse.ArgumentParser(description="Compute color prototypes from sender output.")
    parser.add_argument("file_path", type=str, help="Path to the sender output file.")
    parser.add_argument("--pickle_output", type=str, default="prototypes.pkl", help="Output path for the pickle file.")
    parser.add_argument("--csv_output", type=str, default="prototypes.csv", help="Output path for the CSV file.")
    parser.add_argument("--rl", action="store_true", help="Use RL parsing mode.")
    parser.add_argument("--use_label", action="store_true", help="Use label instead of agent color name.")

    args = parser.parse_args()

    color_data_by_cond, words_by_cond, all_color_data, all_words_used = extract_color_data(
        args.file_path, rl=args.rl, use_label=args.use_label
    )

    compute_prototypes(all_color_data, args.file_path, args.pickle_output, args.csv_output)
    # visualize_prototypes(args.pickle_output)

if __name__ == "__main__":
    main()


