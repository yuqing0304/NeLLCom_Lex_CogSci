import os
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import re
import argparse


# python 3_freq_human.py '../condition1_human/dump_context/msg_spk_seed111'


def run_color_analysis_human(seeds_directories):
    """Run the entire color analysis process."""
    epoch_color_names = aggregate_color_name_counts_human(seeds_directories)
    plot_epoch_histograms_human(epoch_color_names)


def aggregate_color_name_counts_human(seeds_directories):
    """Aggregate color name counts for each epoch from different seed directories."""
    epoch_color_names = {}

    for directory in seeds_directories:
        # print(f"Processing seed directory: {directory}")

        # Get all valid sender_epochX.txt files
        for file_name in sorted(os.listdir(directory)):
            epoch = extract_epoch_from_filename(file_name)
            if epoch is not None:
                file_path = os.path.join(directory, file_name)
                data = process_file_human(file_path)
                color_names = accumulate_color_names(data)

                # Initialize counter if not present
                if epoch not in epoch_color_names:
                    epoch_color_names[epoch] = Counter()

                # Update the counter
                epoch_color_names[epoch].update(color_names)

                # print(f"Accumulated {len(color_names)} color names for epoch {epoch}")
    
    return epoch_color_names




def extract_epoch_from_filename(file_name):
    """Extract the epoch number from the filename, handling both 'output_epoch' and 'sender_epoch' formats."""
    try:
        if "output_epoch" in file_name:
            epoch_str = file_name.split('output_epoch')[1].split('.txt')[0]
        elif "sender_epoch" in file_name:
            epoch_str = file_name.split('sender_epoch')[1].split('.txt')[0]
        else:
            print(f"Warning: No valid epoch identifier found in filename: {file_name}")
            return None  # No valid identifier found

        if not epoch_str.isdigit():
            print(f"Warning: Extracted non-numeric epoch '{epoch_str}' from {file_name}")
            return None  # Ensure the extracted epoch is numeric

        return epoch_str  # Return the extracted epoch number as a string

    except Exception as e:
        print(f"Error extracting epoch from {file_name}: {e}")
        return None  # Return None if extraction fails




def process_file_human(file_path):
    """Read and process a file to extract color names."""
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split("->")
                if len(parts) < 2:
                    continue
                hls_str, color_name = parts[0].strip(), parts[1].strip()
                color_name = color_name.split("label=")[-1].strip(" )") if "label=" in color_name else color_name
                data.append([hls_str, color_name])
        # print(f"Processed {len(data)} lines from {file_path}")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    return data


def accumulate_color_names(data):
    """Accumulate color names from the data"""
    color_names = []
    for entry in data:
        color_name = entry[1]
        color_names.append(color_name)
    return color_names



def plot_epoch_histograms_human(epoch_color_names):
    """Plot and save frequency histograms for each epoch with a fixed x-axis order."""
    # id_to_colors = {i: color for i, color in enumerate([
    #     'mustard', 'cyan', 'maroon', 'lavander', 'medium', 'blood', 'turquoise', 'purple', 
    #     'blue', 'grapes', 'caca', 'teal', 'sky', 'grass', 'red', 'seafoam', 'aqua', 'clay', 
    #     'barney', 'green', 'concrete', 'pumpkin', 'drab', 'tan', 'neon', 'olive', 'lavender', 
    #     'fuchsia', 'gray', 'magenta', 'grape', 'dull', 'peach', 'mint', 'mauve', 'yellow', 
    #     'sage', 'brown', 'pink', 'beige', 'gold', 'orange', 'salmon', 'bright', 'seagreen', 
    #     'violet', 'khaki', 'rose', 'lime'
    # ])}

    # fixed_color_order = list(id_to_colors.values())

    # ======== filter unccessful trials =========
    # sorted_id_to_colors = {8: 'blue', 7: 'purple', 19: 'green', 28: 'gray', 38: 'pink', 14: 'red', 35: 'yellow', 41: 'orange', 37: 'brown', 11: 'teal', 23: 'tan', 16: 'aqua', 25: 'olive', 12: 'sky', 13: 'grass', 6: 'turquoise', 24: 'neon', 40: 'gold', 48: 'lime', 34: 'mauve', 29: 'magenta', 0: 'mustard', 31: 'dull', 18: 'barney', 43: 'bright', 45: 'violet', 47: 'rose', 39: 'beige', 26: 'lavender', 20: 'concrete', 42: 'salmon', 27: 'fuchsia', 21: 'pumpkin', 2: 'maroon', 30: 'grape', 36: 'sage', 10: 'caca', 3: 'lavander', 5: 'blood', 1: 'cyan', 15: 'seafoam', 17: 'clay', 44: 'seagreen', 46: 'khaki', 9: 'grapes', 22: 'drab', 33: 'mint', 4: 'medium', 32: 'peach'}
    # ======== filter unccessful trials =========

    # sorted_id_to_colors = {8: 'blue', 7: 'purple', 28: 'gray', 19: 'green', 38: 'pink', 14: 'red', 35: 'yellow', 41: 'orange', 37: 'brown', 11: 'teal', 23: 'tan', 16: 'aqua', 25: 'olive', 12: 'sky', 48: 'lime', 6: 'turquoise', 13: 'grass', 34: 'mauve', 24: 'neon', 40: 'gold', 29: 'magenta', 0: 'mustard', 45: 'violet', 18: 'barney', 31: 'dull', 43: 'bright', 47: 'rose', 26: 'lavender', 39: 'beige', 27: 'fuchsia', 42: 'salmon', 20: 'concrete', 1: 'cyan', 10: 'caca', 30: 'grape', 21: 'pumpkin', 3: 'lavander', 2: 'maroon', 36: 'sage', 15: 'seafoam', 17: 'clay', 5: 'blood', 9: 'grapes', 44: 'seagreen', 46: 'khaki', 4: 'medium', 22: 'drab', 32: 'peach', 33: 'mint'}

    id_to_colors = {
        0: 'mustard', 1: 'cyan', 2: 'maroon', 3: 'lavander', 4: 'medium', 5: 'blood', 6: 'turquoise', 7: 'purple',
        8: 'blue', 9: 'grapes', 10: 'caca', 11: 'teal', 12: 'sky', 13: 'grass', 14: 'red', 15: 'seafoam', 16: 'aqua',
        17: 'clay', 18: 'barney', 19: 'green', 20: 'concrete', 21: 'pumpkin', 22: 'drab', 23: 'tan', 24: 'neon',
        25: 'olive', 26: 'lavender', 27: 'fuchsia', 28: 'gray', 29: 'magenta', 30: 'grape', 31: 'dull', 32: 'peach',
        33: 'mint', 34: 'mauve', 35: 'yellow', 36: 'sage', 37: 'brown', 38: 'pink', 39: 'beige', 40: 'gold',
        41: 'orange', 42: 'salmon', 43: 'bright', 44: 'seagreen', 45: 'violet', 46: 'khaki', 47: 'rose', 48: 'lime'
    }

    # fixed_color_order = list(sorted_id_to_colors.values())

    first_epoch = next(iter(epoch_color_names))  # Get the first epoch
    color_counts = epoch_color_names[first_epoch]  # Get the corresponding color counts
    full_color_counts = {color: 0 for color in id_to_colors.values()}
    full_color_counts.update(color_counts)
    print(len(full_color_counts))

    # Sort colors by frequency (descending order)
    sorted_colors = sorted(full_color_counts.items(), key=lambda x: x[1], reverse=True)
    print(sorted_colors)
    sorted_color_names, sorted_frequencies = zip(*sorted_colors)

    # Assign colors: light gray for zero count, sky blue for nonzero
    bar_colors = ['lightgray' if count == 0 else 'skyblue' for count in full_color_counts.values()]

    plt.figure(figsize=(12, 6))
    plt.bar(sorted_color_names, sorted_frequencies, color='skyblue')
    plt.xlabel('Color Name')
    plt.ylabel('Frequency')
    # plt.title(f'Color Name Frequency for Epoch {first_epoch}')
    plt.title(f'Human production')
    plt.xticks(rotation=90)
    plt.show()

    output_plot_file = f'epoch{first_epoch}_color_freq_human.png'
    plt.savefig(output_plot_file)
    plt.close()
    print(f"Saved histogram for Epoch {first_epoch}")

    # for epoch, color_counts in epoch_color_names.items():
    #     complete_counts = {color: color_counts.get(color, 0) for color in fixed_color_order}

    #     plt.figure(figsize=(12, 6))
    #     plt.bar(complete_counts.keys(), complete_counts.values(), color='skyblue')
    #     plt.xlabel('Color Name')
    #     plt.ylabel('Frequency')
    #     plt.title(f'Color Name Frequency for Epoch {epoch}')
    #     plt.xticks(rotation=80, fontsize=12)
    #     plt.tight_layout()

    #     output_plot_file = f'epoch{epoch}_color_freq_human.png'
    #     plt.savefig(output_plot_file)
    #     plt.close()
    #     print(f"Saved histogram for Epoch {epoch}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze color production frequencies from human data.")
    parser.add_argument("directories", nargs='+', help="List of directories containing seed files.")
    args = parser.parse_args()
    run_color_analysis_human(args.directories)