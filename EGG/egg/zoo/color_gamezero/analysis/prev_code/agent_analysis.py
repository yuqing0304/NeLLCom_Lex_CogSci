import os
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import re



#=========================================frequency of the 3000 human data by human speakers============================================


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

def run_color_analysis_human(seeds_directories):
    """Run the entire color analysis process."""
    epoch_color_names = aggregate_color_name_counts_human(seeds_directories)
    plot_epoch_histograms_human(epoch_color_names)


#=========================================frequency and entropy============================================

def accumulate_color_names(data):
    """Accumulate color names from the data"""
    color_names = []
    for entry in data:
        color_name = entry[1]
        color_names.append(color_name)
    return color_names


def process_file_freq(file_path):
    """Process the file and extract color names, skipping unsuccessful trials."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split("->")
            if len(parts) < 5:  
                continue  # Skip if the line doesn't have enough parts

            hls_str = parts[0].strip()
            color_name = parts[1].strip()
            outcome_1 = parts[3].strip()
            outcome_2 = parts[4].strip()

            if outcome_1 != outcome_2:
                continue  # Skip unsuccessful trials
            data.append([hls_str, color_name])

    return data


def process_file_freq_spk(file_path):
    """Process the file and extract color names, skipping unsuccessful trials."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split("->")
            if len(parts) < 2:  
                continue  # Skip if the line doesn't have enough parts

            hls_str = parts[0].strip()
            # Extract color name (before label=)
            color_part = parts[1].strip()
            match = re.search(r"(.+?)\s*\(label=(\w+)\)", color_part)
            if not match:
                continue  # Skip if format is incorrect
            
            color_name = match.group(1).strip()  # Extracted color name
            label_color = match.group(2).strip()  # Extracted label color

            if color_name != label_color:
                continue  # Skip if label color doesn't match extracted color name

            data.append([hls_str, color_name])

    return data


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


def rename_files(directory):
    """Rename the output files for each epoch"""
        
    files_to_rename = [('output_Initial Eval.txt', 'output_epoch0.txt')] + [
        (f'output_{i}.txt', f'output_epoch{i+1}.txt') for i in range(30)
    ]

    # Rename each file in the list
    for old_file_name, new_file_name in files_to_rename:
        old_file_path = os.path.join(directory, old_file_name)
        
        if os.path.exists(old_file_path):
            new_file_path = os.path.join(directory, new_file_name)
            os.rename(old_file_path, new_file_path)
            # print(f"Renamed {old_file_name} to {new_file_name}")
        # else:
        #     print(f"File {old_file_name} not found in {directory}")



def aggregate_color_name_counts(seeds_directories):
    """Aggregate color name counts across different seeds for each epoch"""
    epoch_color_names = {str(i): Counter() for i in range(31)}  # Assuming epochs 0 to 30

    # Iterate over each seed directory
    for directory in seeds_directories:
        print(f"Processing seed directory: {directory}")

        # Determine whether it's "spk" or "rf"
        if "spk" in directory:
            process_function = process_file_freq_spk  # Use the spk version
            files = [f'{directory}/sender_epoch{i}.txt' for i in range(30)]
        elif "rf" in directory:
            process_function = process_file_freq  # Use the rf version
            # List of renamed files to process for each seed (epoch 0 to 30)
            files = [f'{directory}/output_epoch{i}.txt' for i in range(31)]
        else:
            print(f"Warning: Unknown directory type in {directory}. Skipping...")
            continue  # Skip unknown directories

        # Iterate over the files for this seed
        for file in files:
            epoch = extract_epoch_from_filename(file)
            if epoch:
                data = process_function(file)  # Call the chosen function
                color_names = accumulate_color_names(data)
                
                # Aggregate color name counts for this epoch across seeds
                epoch_color_names[epoch].update(color_names)
            else:
                print(f"Warning: Could not extract epoch from file {file}")

    return epoch_color_names


def calculate_entropy_based_on_word_frequency(color_counts):
    """Calculate the entropy based on word frequency (color name frequency) using the Shannon formula."""
    total_count = sum(color_counts.values())
    if total_count == 0:
        return 0  # Return 0 if there are no color names
    
    # Convert color counts to probabilities
    probabilities = np.array(list(color_counts.values())) / total_count

    # Remove zero probabilities to avoid log2(0) issues
    probabilities = probabilities[probabilities > 0]

    # Calculate entropy using the Shannon formula: H(X) = -sum(p * log2(p))
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy_value


def plot_epoch_histograms(epoch_color_names, specific_epoch=None):
    """
    Plot and save frequency histograms for each epoch with a fixed x-axis order.
    If specific_epoch is provided, plot only that epoch.
    """
    # id_to_colors = {0: 'mustard', 1: 'cyan', 2: 'maroon', 3: 'lavander', 4: 'medium', 5: 'blood', 6: 'turquoise', 7: 'purple', 8: 'blue', 9: 'grapes', 
    #                 10: 'caca', 11: 'teal', 12: 'sky', 13: 'grass', 14: 'red', 15: 'seafoam', 16: 'aqua', 17: 'clay', 18: 'barney', 19: 'green', 20: 'concrete', 
    #                 21: 'pumpkin', 22: 'drab', 23: 'tan', 24: 'neon', 25: 'olive', 26: 'lavender', 27: 'fuchsia', 28: 'gray', 29: 'magenta', 30: 'grape', 31: 'dull', 
    #                 32: 'peach', 33: 'mint', 34: 'mauve', 35: 'yellow', 36: 'sage', 37: 'brown', 38: 'pink', 39: 'beige', 40: 'gold', 41: 'orange', 42: 'salmon', 
    #                 43: 'bright', 44: 'seagreen', 45: 'violet', 46: 'khaki', 47: 'rose', 48: 'lime'}

    # fixed_color_order = list(id_to_colors.values())

    # ordered_colors = {8: 'blue', 7: 'purple', 19: 'green', 28: 'gray', 38: 'pink', 14: 'red', 35: 'yellow', 41: 'orange', 
    #                   37: 'brown', 11: 'teal', 23: 'tan', 16: 'aqua', 25: 'olive', 12: 'sky', 13: 'grass', 6: 'turquoise', 
    #                   24: 'neon', 40: 'gold', 48: 'lime', 34: 'mauve', 29: 'magenta', 0: 'mustard', 31: 'dull', 18: 'barney', 
    #                   43: 'bright', 45: 'violet', 47: 'rose', 39: 'beige', 26: 'lavender', 20: 'concrete', 42: 'salmon', 
    #                   27: 'fuchsia', 21: 'pumpkin', 2: 'maroon', 30: 'grape', 36: 'sage', 10: 'caca', 3: 'lavander', 5: 'blood', 
    #                   1: 'cyan', 15: 'seafoam', 17: 'clay', 44: 'seagreen', 46: 'khaki', 9: 'grapes', 22: 'drab', 33: 'mint', 
    #                   4: 'medium', 32: 'peach'}

    ordered_colors = {
    1: 'blue', 2: 'purple', 3: 'green', 4: 'gray', 5: 'pink', 
    6: 'yellow', 7: 'red', 8: 'orange', 9: 'brown', 10: 'teal', 
    11: 'tan', 12: 'aqua', 13: 'olive', 14: 'mauve', 15: 'magenta', 
    16: 'sky', 17: 'gold', 18: 'neon', 19: 'turquoise', 20: 'lime', 
    21: 'grass', 22: 'violet', 23: 'barney', 24: 'lavender', 25: 'beige', 
    26: 'grape', 27: 'bright', 28: 'mustard', 29: 'maroon', 30: 'grapes', 
    31: 'dull', 32: 'sage', 33: 'khaki', 34: 'lavander', 35: 'caca', 
    36: 'concrete', 37: 'seagreen', 38: 'cyan', 39: 'blood', 40: 'seafoam', 
    41: 'drab', 42: 'medium', 43: 'clay', 44: 'fuchsia', 45: 'peach', 
    46: 'salmon', 47: 'rose', 48: 'pumpkin', 49: 'mint'
}


    fixed_color_order = list(ordered_colors.values())

    if specific_epoch is not None:
        # Plot only the specific epoch
        epoch_key = str(specific_epoch)
        if epoch_key not in epoch_color_names:
            print(f"Warning: Epoch {specific_epoch} not found in the data.")
            return
        
        color_counts = epoch_color_names[epoch_key]
        ### process the parenthesis begin
        # Dictionary to store summed values
        summed_colors = defaultdict(int)

        # Process the dictionary to remove content in brackets and sum values
        for key, value in color_counts.items():
            clean_key = re.sub(r"\s*\(.*?\)", "", key)  # Remove parentheses and content
            summed_colors[clean_key] += value  # Sum frequencies

        # Convert defaultdict back to a regular dictionary
        summed_colors = dict(summed_colors)


        num_successful_trials = sum(summed_colors.values())  # Total count of all colors
        print(f"The number of successful trials in epoch {specific_epoch}: {num_successful_trials}")

        complete_counts = {color: summed_colors.get(color, 0) for color in fixed_color_order}
        
        plt.figure(figsize=(10, 5))
        plt.bar(complete_counts.keys(), complete_counts.values(), color='skyblue')
        plt.xlabel('Color Name')
        plt.ylabel('Frequency')
        # plt.ylim(0, 4000)
        plt.title(f'Epoch {specific_epoch}')
        plt.xticks(rotation=80)
        plt.tight_layout()
        plt.show()
    
    else:
        # Plot all epochs in a 4x8 grid
        fig, axes = plt.subplots(8, 4, figsize=(16, 30))
        axes = axes.flatten()

        for idx, (epoch, color_counts) in enumerate(epoch_color_names.items()):
            complete_counts = {color: color_counts.get(color, 0) for color in fixed_color_order}

            ax = axes[idx]
            ax.bar(complete_counts.keys(), complete_counts.values(), color='skyblue')
            ax.set_xlabel('Color Name')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Epoch {epoch}')
            ax.tick_params(axis='x', rotation=80)

        plt.tight_layout()
        plt.show()


def freq_dist_agent(seeds_directories, plot_all=True, specific_epoch=None):
    """
    Process color frequency data and plot histograms.
    
    Parameters:
    - seeds_directories: List of directories containing seed data
    - plot_all (bool): If True, plots histograms for all epochs
    - specific_epoch (int or None): If provided, plots only that epoch
    """
    for dir in seeds_directories:
        rename_files(dir)

    epoch_color_names = aggregate_color_name_counts(seeds_directories)

    # Choose the appropriate plotting behavior
    if specific_epoch is not None:
        plot_epoch_histograms(epoch_color_names, specific_epoch=specific_epoch)
    elif plot_all:
        plot_epoch_histograms(epoch_color_names)



def plot_entropy_over_epochs(seeds_directories):
    """Plot entropy as a function of epoch with a minimal style."""
    epoch_color_names = aggregate_color_name_counts(seeds_directories)
    epochs = sorted(epoch_color_names.keys(), key=lambda x: int(x))  # Sort epochs numerically
    entropies = [calculate_entropy_based_on_word_frequency(epoch_color_names[epoch]) for epoch in epochs]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, entropies, marker='o', linestyle='-', color='black', markersize=5, linewidth=1)

    plt.axhline(y=3.6346, color='red', linestyle='--', linewidth=1, label='Entropy of human production')
    # Minimalist styling
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Entropy', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title('Entropy Over Epochs', fontsize=14)
    plt.ylim(0, max(entropies) + 0.8)  # Ensure y-axis starts above 0
    plt.gca().spines[['top', 'right']].set_visible(False)  # Remove top/right borders
    plt.gca().spines[['left', 'bottom']].set_linewidth(0.8)  # Thin axis lines
    plt.show()



#=========================================word informativeness============================================


import pandas as pd
import random
import numpy as np
import ast
import os
import colorsys
import colorspacious as cs
import matplotlib.pyplot as plt


# Function to convert HLS to RGB
def hls_to_rgb(h, l, s):
    h = h / 360.0  # Scale hue to [0, 1]
    l = l / 100.0  # Scale lightness to [0, 1]
    s = s / 100.0  # Scale saturation to [0, 1]
    return tuple(round(i * 255) for i in colorsys.hls_to_rgb(h, l, s))

# Function to convert RGB to CIELAB
def rgb_to_cielab(r, g, b):
    return cs.cspace_convert((r, g, b), start="sRGB255", end="CIELab")

# Function to calculate CIELAB color distance
def compute_cielab_distance(color1, color2):
    return np.linalg.norm(np.array(color1) - np.array(color2))

def accumulate_tar_cielab(data):
    # Dictionary to store accumulated HLS values for each color name
    color_cielab_dict = {}

    for entry in data:
        # Extract the first HLS values and the color name from the line
        hls_values = entry[0]  # First HLS value from the list
        color_name = entry[1]

        # Convert HLS value to RGB and then to CIELAB
        # rgb_color = hls_to_rgb(hls_values[0], hls_values[1], hls_values[2])
        # cielab_color = rgb_to_cielab(rgb_color[0], rgb_color[1], rgb_color[2])
        #### to correct!!!!!!
        rgb_color = hls_to_rgb(hls_values[0], hls_values[1], hls_values[2])
        cielab_color = rgb_to_cielab(*rgb_color)

        # Accumulate the CIELAB values for each color name
        if color_name not in color_cielab_dict:
            color_cielab_dict[color_name] = []
        
        color_cielab_dict[color_name].append(cielab_color)

    return color_cielab_dict


def compute_informativeness_for_words(color_hls_dict):
    informativeness = {}

    for word, cols in color_hls_dict.items():
        tmp = []
        if len(cols) < 100:
            for c in cols:
                for c2 in cols:
                    distance = compute_cielab_distance(c, c2)
                    if distance != 0.0:  # Only add non-zero distances
                        tmp.append(distance)
        else:
            tmp2 = []
            for i in range(30):  # Sample 100 pairs if there are more than 100 colors
                sampled = random.sample(cols, 100)
                for c in sampled:
                    for c2 in sampled:
                        distance = compute_cielab_distance(c, c2)
                        if distance != 0.0:  # Only add non-zero distances
                            tmp2.append(distance)
            if tmp2:  # Only calculate if there are non-zero distances
                tmp.append(sum(tmp2) / len(tmp2))
        
        # Only compute informativeness if there are valid distances (i.e., tmp is not empty)
        if tmp:
            tmp = [i for i in tmp if i != 0.0]  # Remove any remaining zeros
            if tmp:  # Ensure tmp is not empty after removing zeros
                informativeness[word] = 100 / (sum(tmp) / len(tmp))
            else:
                informativeness[word] = 0  # Set to 0 if no valid distances were found
        else:
            informativeness[word] = 0  # Set to 0 if no valid distances were computed

    return informativeness


def process_files(file_paths, output_file_path):
    # Open the output file in write mode
    with open(output_file_path, 'w') as output_file:
        # Iterate through each file in the provided list of file paths
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                # Read the content of each file and write it to the output file
                output_file.write(f.read())  # Write the entire content of the file
    
    # print(f"All files have been concatenated and saved to {output_file_path}")


def process_file_info(file_path):
    # Read data from the given file
    n = 0
    with open(file_path, 'r') as f:
        data = []
        for line in f:
            parts = line.strip().split("->")
            
            # Extract HLS values and color name
            hls_str = parts[0].strip()
            hls_values = ast.literal_eval(hls_str)  # Convert string to list of HLS values
            color_name = parts[1].strip()  # Color name comes after '->'

            outcome_1 = parts[3].strip() 
            # print(f"outcome_1, {outcome_1}")
            outcome_2 = parts[4].strip() 
            condition = parts[5].strip()

            # Check if outcome is True
            if outcome_1 != outcome_2:
                n = n+1
                continue  # Skip unsuccessful trials

            # Append to data (we take only the first HLS value)
            data.append([hls_values[0], color_name])  # Only first HLS value is considered

    # print(f"n, {n}")
    # print(f"len, {len(data)}")

    # Step 1: Accumulate all the HLS values for each color name
    color_cielab_dict = accumulate_tar_cielab(data)

    # Step 2: Compute informativeness for the current file
    informativeness = compute_informativeness_for_words(color_cielab_dict)

    # Convert informativeness to DataFrame for easy viewing and saving
    informativeness_df = pd.DataFrame(list(informativeness.items()), columns=['color_name', 'informativeness']).sort_values(by='informativeness', ascending=False)

    # Save the informativeness to a CSV for the current file to track changes
    output_file = file_path.replace('.txt', '_informativeness.csv')
    informativeness_df.to_csv(output_file, index=False)

    # Plot informativeness distribution
    plot_informativeness_distribution(informativeness_df, file_path)

    return informativeness_df


def plot_informativeness_distribution(informativeness_df, file_path):
    # Plot informativeness distribution for color names
    plt.figure(figsize=(10, 6))
    plt.bar(informativeness_df['color_name'], informativeness_df['informativeness'], color='skyblue')
    plt.xlabel('Color Name')
    plt.ylabel('Informativeness')
    plt.title(f'Informativeness Distribution for {os.path.basename(file_path)}')
    plt.xticks(rotation=80, fontsize=16)
    plt.tight_layout()

    # Save the plot as a PNG file
    output_plot_file = file_path.replace('.txt', '_informativeness_distribution.png')
    plt.savefig(output_plot_file)
    plt.close()


def word_informativeness(file_paths, output_file_path):

    process_files(file_paths, output_file_path)

    # Process each file and compute informativeness independently
    # for file in files:
    print(f"Processing file: {output_file_path}")
    informativeness_df = process_file_info(output_file_path)

    # Print informativeness for the current file
    print(f"Informativeness for {output_file_path}:")
    print(informativeness_df)



#=========================================correlate word informativeness============================================

# computed_informativeness_file = "../dump_context/epoch30_informativeness.csv"
# reference_file = "../dynamics/language_use/informativeness/dic_informativeness.csv"
# correlate_word_informativeness(computed_informativeness_file, reference_file, method="pearson") # --- spearman correlation

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def correlate_word_informativeness(computed_informativeness_file, reference_file, method):
    # Read the reference informativeness file
    reference_df = pd.read_csv(reference_file)

    # Read the computed informativeness file
    computed_df = pd.read_csv(computed_informativeness_file)

    # Merge both dataframes on color_name, keeping only rows with matching color names
    merged_df = computed_df.merge(reference_df, on="color_name", suffixes=("_computed", "_reference"))

    # Remove rows where either informativeness value is zero
    merged_df = merged_df[(merged_df["informativeness_computed"] > 0) & (merged_df["informativeness_reference"] > 0)]

    # Compute correlation
    # correlation = merged_df["informativeness_computed"].corr(merged_df["informativeness_reference"])
    if method == "spearman":
        correlation = merged_df["informativeness_computed"].corr(merged_df["informativeness_reference"], method="spearman")
    elif method == "pearson":
        correlation = merged_df["informativeness_computed"].corr(merged_df["informativeness_reference"], method="pearson")
    print(f"{method} correlation: {correlation:.4f}")

    # Scatterplot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=merged_df, x="informativeness_computed", y="informativeness_reference", alpha=0.7)

    # Add labels to the scatterplot
    for i in range(merged_df.shape[0]):
        plt.text(merged_df["informativeness_computed"].iloc[i],
                 merged_df["informativeness_reference"].iloc[i],
                 merged_df["color_name"].iloc[i],
                 fontsize=9, ha='right', va='bottom')

    # Set the same range for both axes
    min_val = min(merged_df["informativeness_computed"].min(), merged_df["informativeness_reference"].min())
    max_val = max(merged_df["informativeness_computed"].max(), merged_df["informativeness_reference"].max())
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    plt.xlabel("Computed Informativeness")
    plt.ylabel("Reference Informativeness")
    plt.title(f"Scatterplot of Informativeness (Correlation: {correlation:.2f})")
    plt.grid(True)
    plt.show()


#=========================================color prototype============================================

import ast
import pickle
import numpy as np
import colorspacious as cs
import csv
from collections import defaultdict

def rgb_to_cielab(r, g, b):
    """Convert RGB (0-255) to CIELAB."""
    return cs.cspace_convert((r, g, b), start="sRGB255", end="CIELab")

def process_myfile(file_path):
    """Process the file and extract color names and RGB values, skipping unsuccessful trials."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split("->")
            if len(parts) < 5:
                continue  # Skip if the line doesn't have enough parts

            rgb_str = parts[0].strip()
            color_name = parts[1].strip()
            outcome_1 = parts[3].strip()
            outcome_2 = parts[4].strip()
            condition = parts[5].strip()
            if condition != "close":
                continue 

            if outcome_1 != outcome_2:
                continue  # Skip unsuccessful trials

            try:
                rgb_tuple = ast.literal_eval(rgb_str)  # Convert string "[R, G, B]" to a list
                # print(f"rgb_tuple, {rgb_tuple}")  # Debugging output

                # Ensure it's a valid RGB triplet (list or tuple with 3 elements)
                if isinstance(rgb_tuple, (list, tuple)) and len(rgb_tuple) == 3:
                    data.append((color_name, tuple(rgb_tuple[0])))  # Convert to tuple before storing
            except (SyntaxError, ValueError):
                continue  # Skip malformed RGB entries

    return data

import ast

def process_myfile_spk(file_path):
    """Process the file and extract color names and RGB values, skipping unsuccessful trials."""
    data = []
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split("->")
            if len(parts) < 2:  
                continue  # Skip if the line doesn't have enough parts
            
            rgb_str = parts[0].strip()  # Extract the list of RGB values
            color_info = parts[1].strip()  # Extract the color name part
            
            # Extract the actual color name (e.g., "green") from "green (label=green)"
            color_name = color_info.split("(label=")[-1].replace(")", "").strip()

            try:
                rgb_tuple = ast.literal_eval(rgb_str)  # Convert string "[R, G, B]" to a list
                # print(f"rgb_tuple, {rgb_tuple}")  # Debugging output

                # Ensure it's a valid RGB triplet (list or tuple with 3 elements)
                if isinstance(rgb_tuple, (list, tuple)) and len(rgb_tuple) == 3:
                    data.append((color_name, tuple(rgb_tuple[0])))  # Convert to tuple before storing
            except (SyntaxError, ValueError):
                continue  # Skip malformed RGB entries
            
    return data


def compute_prototypes(data):
    """Compute the prototype (mean CIELAB value) for each color name."""
    color_dict = defaultdict(list)

    # Convert RGB to CIELAB and group by color name
    for color_name, hls in data:
        rgb = hls_to_rgb(*hls)
        color_dict[color_name].append(rgb_to_cielab(*rgb))

    # Compute the mean prototype for each color name
    prototypes = {color: np.mean(np.array(lab_values), axis=0) for color, lab_values in color_dict.items()}
    
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


def compute_prototype(input_file, pickle_output, csv_output):
    """Main function to process the file, compute prototypes, and save results."""
    
    # Determine which processing function to use
    if "spk" in input_file:
        data = process_myfile_spk(input_file)
    elif "rf" in input_file:
        data = process_myfile(input_file)
    else:
        raise ValueError("Unknown file type: input_file should contain 'spk' or 'rf' to specify the format.")

    # Compute and save prototypes
    prototypes = compute_prototypes(data)
    save_prototypes_pickle(prototypes, pickle_output)
    save_prototypes_csv(prototypes, csv_output)



#=========================================visualize prototype============================================



import pickle
import numpy as np
import colorspacious as cs
import matplotlib.pyplot as plt

def load_prototypes(pickle_path):
    """Load color prototypes from a pickle file."""
    with open(pickle_path, "rb") as f:
        prototypes = pickle.load(f)
    return prototypes

def cielab_to_rgb(lab):
    """Convert CIELAB to sRGB (0-1 range)."""
    rgb = cs.cspace_convert(lab, start="CIELab", end="sRGB1")
    rgb = np.clip(rgb, 0, 1)  # Ensure values are within valid RGB range
    return rgb

def plot_color_prototypes(prototypes):
    """Visualize color prototypes in a grid."""
    fig, ax = plt.subplots(figsize=(50, 4))
    
    # Sort colors alphabetically for consistency
    color_names = sorted(prototypes.keys())

    for i, color in enumerate(color_names):
        lab_value = prototypes[color]  # Get CIELAB values
        rgb_value = cielab_to_rgb(lab_value)  # Convert to RGB

        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=rgb_value))  # Draw color block
        ax.text(i + 0.5, -0.3, color, ha='center', va='center', fontsize=14)  # Label

    ax.set_xlim(0, len(color_names))
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)  # Remove borders

    plt.show()

def visualize_prototypes(pickle_path):
    """Load prototypes from a file and visualize them."""
    prototypes = load_prototypes(pickle_path)
    plot_color_prototypes(prototypes)



#=========================a function that computes dist2name and dist2other for each color patch (agent)======================


import pickle
import numpy as np
import colorspacious as cs
import csv

# def load_prototypes(pickle_path):
#     """Load color prototypes from a pickle file."""
#     with open(pickle_path, "rb") as f:
#         return pickle.load(f)


# # Function to calculate CIELAB color distance
# def compute_cielab_distance(color1, color2):
#     return np.linalg.norm(np.array(color1) - np.array(color2))


def compute_distances(data, prototypes):
    """
    Compute:
    - dist2name: distance from target color to its assigned prototype.
    - dist2other: distance from target color to the nearest alternative prototype.
    
    Returns:
    - A list of results containing (color_name, LAB, dist2name, dist2other).
    - The average dist2name and dist2other across all data points.
    """
    results = []
    total_dist2name = 0
    total_dist2other = 0
    count = 0

    for color_name, hls in data:
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
        else:
            dist2other = None  # No other prototypes to compare

        results.append((color_name, target_lab, dist2name, dist2other))

        # Accumulate for average calculation
        total_dist2name += dist2name
        if dist2other is not None:
            total_dist2other += dist2other
        count += 1

    # Compute averages
    avg_dist2name = total_dist2name / count if count > 0 else None
    avg_dist2other = total_dist2other / count if count > 0 else None

    return results, avg_dist2name, avg_dist2other

def save_distance_results(results, avg_dist2name, avg_dist2other, output_csv):
    """Save distance calculations to a CSV file for inspection and print averages."""
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["color_name", "L", "A", "B", "dist2name", "dist2other"])  # Header
        for name, lab, d2n, d2o in results:
            writer.writerow([name] + list(lab) + [d2n, d2o])
    
    print(f"Results saved successfully to {output_csv}!")
    print(f"Average dist2name: {avg_dist2name:.3f}")
    print(f"Average dist2other: {avg_dist2other:.3f}")

def analyze_typicality(data_file, prototypes_file, output_csv):
    """
    Compute typicality measures (dist2name & dist2other), save the results, and print averages.
    """
    prototypes = load_prototypes(prototypes_file)
    data = process_myfile(data_file)  # Load data from file
    results, avg_dist2name, avg_dist2other = compute_distances(data, prototypes)  # Compute distances
    save_distance_results(results, avg_dist2name, avg_dist2other, output_csv)  # Save & print results



#=========================a function that computes dist2name and dist2other for each color patch (human)======================


def compute_distances_human(data, prototypes):
    """
    Compute:
    - dist2name: distance from target color to its assigned prototype.
    - dist2other: distance from target color to the nearest alternative prototype.
    
    Returns:
    - A list of results containing (color_name, LAB, dist2name, dist2other).
    - The average dist2name and dist2other across all data points.
    """
    results = []
    total_dist2name = 0
    total_dist2other = 0
    count = 0

    for target_lab, color_name in data:
        # Prototype lab for the given color name
        prototype_lab = prototypes.get(color_name, None)

        if prototype_lab is None:
            continue  # Skip if no prototype exists for this color name

        # Compute dist2name (distance from target color to its assigned prototype)
        dist2name = compute_cielab_distance(target_lab, prototype_lab)

        # Compute dist2other (distance from target color to the nearest alternative prototype)
        other_prototypes = {name: lab for name, lab in prototypes.items() if name != color_name}
        if other_prototypes:
            dist2other = min(compute_cielab_distance(target_lab, other_lab) for other_lab in other_prototypes.values())
        else:
            dist2other = None  # No other prototypes to compare

        results.append((color_name, target_lab, dist2name, dist2other))

        # Accumulate for average calculation
        total_dist2name += dist2name
        if dist2other is not None:
            total_dist2other += dist2other
        count += 1

    # Compute averages
    avg_dist2name = total_dist2name / count if count > 0 else None
    avg_dist2other = total_dist2other / count if count > 0 else None

    return results, avg_dist2name, avg_dist2other


def process_human_data(file_path):
    """Process the human data file into a list of target CIELAB colors and their labels, skipping the first line."""
    data = []
    with open(file_path, "r") as f:
        next(f)  # Skip the first line
        for line in f:
            parts = line.strip().split(",", 1)  # Split at the first comma
            if len(parts) != 2:
                continue  # Skip malformed lines
            # print(f"Processing line: {line}")  # Debug print
            element = line.strip().split(",")

            # Split the CIELAB values (parts[0]) by whitespace and convert to floats
            target_lab = list(map(float, parts[0].strip("[]").split()))  # Convert string to list of floats
            color_name = element[3].strip()  # Extract the color name
            # print(target_lab, color_name)
            data.append((target_lab, color_name))
    return data


def analyze_typicality_human(data_file, prototypes_file, output_csv):
    """
    Compute typicality measures (dist2name & dist2other) for human data, save the results, and print averages.
    """
    prototypes = load_prototypes(prototypes_file)
    data = process_human_data(data_file)  # Load data from the human file
    results, avg_dist2name, avg_dist2other = compute_distances_human(data, prototypes)  # Compute distances
    save_distance_results(results, avg_dist2name, avg_dist2other, output_csv)  # Save & print results




    #================================informativeness diff===========================================

import ast
import pandas as pd
import argparse

def process_file_infodiff(file_path, info_path):
    """
    Processes the input file to extract color informativeness difference between 'far' and 'close' conditions.
    
    Parameters:
        file_path (str): Path to the input text file.
        info_path (str): Path to the informativeness CSV file.

    Returns:
        pd.DataFrame: Informativeness DataFrame.
        float: Difference in informativeness (Far - Close).
    """
    condition_dict = {"far": [], "close": []}  # Track words used per condition
    
    # Read and process the input text file
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split("->")
            if len(parts) < 6:
                continue  # Skip malformed lines
            
            hls_str, color_name = parts[0].strip(), parts[1].strip()
            outcome_1, outcome_2, condition = parts[3].strip(), parts[4].strip(), parts[5].strip()
            
            if outcome_1 != outcome_2:
                continue  # Skip unsuccessful trials
            
            if condition in condition_dict:
                condition_dict[condition].append(color_name)
    
    # Read informativeness data
    try:
        informativeness_df = pd.read_csv(info_path)
    except Exception as e:
        print(f"Error reading informativeness file: {e}")
        return None, None
    
    if "color_name" not in informativeness_df.columns or "informativeness" not in informativeness_df.columns:
        print("Error: Informativeness CSV must contain 'color_name' and 'informativeness' columns.")
        return None, None

    # Compute mean informativeness for 'far' and 'close' conditions
    far_words = set(condition_dict["far"])
    close_words = set(condition_dict["close"])
    print(f"far_words, {far_words}")
    print(f"close_words, {close_words}")
    unique_close_words = close_words - far_words
    print(f"Words in close_words but not in far_words: {unique_close_words}")


    mean_far = informativeness_df[informativeness_df["color_name"].isin(far_words)]["informativeness"].mean()
    mean_close = informativeness_df[informativeness_df["color_name"].isin(close_words)]["informativeness"].mean()
    
    informativeness_diff = mean_far - mean_close
    
    print(f"Mean Informativeness (Far): {mean_far:.4f}")
    print(f"Mean Informativeness (Close): {mean_close:.4f}")
    print(f"Informativeness Difference (Far - Close): {informativeness_diff:.4f}")

    return informativeness_df, informativeness_diff

def main():
    parser = argparse.ArgumentParser(description="Compute informativeness difference from a text file.")
    parser.add_argument("file_path", type=str, help="Path to the input text file")
    parser.add_argument("info_path", type=str, help="Path to the informativeness CSV file")
    args = parser.parse_args()
    
    process_file_infodiff(args.file_path, args.info_path)

if __name__ == "__main__":
    main()

#===========================system level informativeness===============================
def process_file_info_system(file_path, word_info_path):
    n = 0
    words_used = []  # Store words that were used
    with open(file_path, 'r') as f:
        data = []
        for line in f:
            parts = line.strip().split("->")
            hls_str = parts[0].strip()
            hls_values = ast.literal_eval(hls_str)
            color_name = parts[1].strip()

            outcome_1 = parts[3].strip()
            outcome_2 = parts[4].strip()
            condition = parts[5].strip()

            # Skip unsuccessful trials
            if outcome_1 != outcome_2:
                n += 1
                continue

            data.append([hls_values[0], color_name])
            words_used.append(color_name)  # Track words used in successful trials

    # Step 1: Accumulate HLS values for each color name
    color_cielab_dict = accumulate_tar_cielab(data)

    # Step 2: Compute informativeness for individual words
    informativeness_df = pd.read_csv(word_info_path)
    informativeness = compute_informativeness_for_words(color_cielab_dict)

    # Step 3: Compute lexical system-level informativeness
    lexical_informativeness = compute_lexical_system_informativeness(informativeness_df, words_used)
    
    # Print results
    print(f"Lexical System Informativeness for {file_path}: {lexical_informativeness}")
    
    return informativeness_df, lexical_informativeness


def compute_lexical_system_informativeness(informativeness_df, words_used):
    """
    Computes the lexical system-level informativeness as the average of word informativeness (Iw) 
    for the words actually used in N interactions.

    Args:
        informativeness_df (pd.DataFrame): DataFrame containing 'color_name' and 'informativeness' columns.
        words_used (list): List of words used in the interactions.

    Returns:
        float: Lexical system informativeness.
    """
    # Filter informativeness values for the words used
    used_informativeness = informativeness_df[informativeness_df["color_name"].isin(words_used)]["informativeness"]
    
    # Compute mean informativeness
    lexical_system_informativeness = used_informativeness.mean()
    
    return lexical_system_informativeness





# =========================a function that computes dist2name and dist2other for each color patch (human)======================


def compute_distances_human(data, prototypes):
    """
    Compute:
    - dist2name: distance from target color to its assigned prototype.
    - dist2other: distance from target color to the nearest alternative prototype.
    
    Returns:
    - A list of results containing (color_name, LAB, dist2name, dist2other).
    - The average dist2name and dist2other across all data points.
    """
    results = []
    total_dist2name = 0
    total_dist2other = 0
    count = 0

    for target_lab, color_name in data:
        # Prototype lab for the given color name
        prototype_lab = prototypes.get(color_name, None)


        if prototype_lab is None:
            continue  # Skip if no prototype exists for this color name

        # Compute dist2name (distance from target color to its assigned prototype)
        dist2name = compute_cielab_distance(target_lab, prototype_lab)

        # Compute dist2other (distance from target color to the nearest alternative prototype)
        other_prototypes = {name: lab for name, lab in prototypes.items() if name != color_name}
        if other_prototypes:
            dist2other = min(compute_cielab_distance(target_lab, other_lab) for other_lab in other_prototypes.values())
        else:
            dist2other = None  # No other prototypes to compare

        results.append((color_name, target_lab, dist2name, dist2other))

        # Accumulate for average calculation
        total_dist2name += dist2name
        if dist2other is not None:
            total_dist2other += dist2other
        count += 1

    # Compute averages
    avg_dist2name = total_dist2name / count if count > 0 else None
    avg_dist2other = total_dist2other / count if count > 0 else None

    return results, avg_dist2name, avg_dist2other


def process_human_data(file_path):
    """Process the human data file into a list of target CIELAB colors and their labels, skipping the first line."""
    data = []
    with open(file_path, "r") as f:
        next(f)  # Skip the first line
        for line in f:
            parts = line.strip().split(",", 1)  # Split at the first comma
            if len(parts) != 2:
                continue  # Skip malformed lines
            # print(f"Processing line: {line}")  # Debug print
            element = line.strip().split(",")

            # Split the CIELAB values (parts[0]) by whitespace and convert to floats
            target_lab = list(map(float, parts[0].strip("[]").split()))  # Convert string to list of floats
            color_name = element[3].strip()  # Extract the color name
            # print(target_lab, color_name)
            data.append((target_lab, color_name))
    return data


def analyze_typicality_human(data_file, prototypes_file, output_csv):
    """
    Compute typicality measures (dist2name & dist2other) for human data, save the results, and print averages.
    """
    prototypes = load_prototypes(prototypes_file)
    data = process_human_data(data_file)  # Load data from the human file
    results, avg_dist2name, avg_dist2other = compute_distances_human(data, prototypes)  # Compute distances
    save_distance_results(results, avg_dist2name, avg_dist2other, output_csv)  # Save & print results
