import os
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import re
import argparse
from collections import defaultdict



# python 3_freq_slrl.py '../condition1_human/dump_context/msg_spk_seed111' --specific_epoch 29
# python 3_freq_slrl.py '../condition1_human/dump_context/msg_rf_seed111' --specific_epoch 30

# python 3_freq_slrl.py '../condition3_generated/dump_context/msg_spk_seed111' --specific_epoch 29
# python 3_freq_slrl.py '../condition3_generated/dump_context/msg_rf_seed111' --specific_epoch 30

# python 3_freq_slrl.py '../condition3_generated/experiment1/dump_context/msg_rf_seed111' --specific_epoch 30


def freq_dist_agent(seeds_directories, specific_epoch=None, plot_all=True):
    """
    Process color frequency data and plot histograms.
    
    Parameters:
    - seeds_directories: List of directories containing seed data
    - plot_all (bool): If True, plots histograms for all epochs
    - specific_epoch (int or None): If provided, plots only that epoch
    """
    for dir in seeds_directories:
        rename_files(dir)

    epoch_color_names, plot_type = aggregate_color_name_counts(seeds_directories)

    # Choose the appropriate plotting behavior
    if specific_epoch is not None:
        plot_epoch_histograms(epoch_color_names, plot_type, specific_epoch=specific_epoch)
    elif plot_all:
        plot_epoch_histograms(epoch_color_names,plot_type)




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
            if "human" in directory:
                plot_type = "spk_human"
            elif "generated" in directory:
                plot_type = "spk_generated"
        elif "rf" in directory:
            process_function = process_file_freq  # Use the rf version
            # List of renamed files to process for each seed (epoch 0 to 30)
            files = [f'{directory}/output_epoch{i}.txt' for i in range(31)]
            if "human" in directory:
                plot_type = "rf_human"
            elif "generated" in directory:
                plot_type = "rf_generated"
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

    return epoch_color_names, plot_type




def plot_epoch_histograms(epoch_color_names, plot_type, specific_epoch=None):
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

        output_plot_file = f'epoch{specific_epoch}_color_freq_{plot_type}.png'
        plt.savefig(output_plot_file)
        plt.close()
        print(f"Saved histogram for Epoch {specific_epoch}")
    
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




def accumulate_color_names(data):
    """Accumulate color names from the data"""
    color_names = []
    for entry in data:
        color_name = entry[1]
        color_names.append(color_name)
    return color_names




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze color production frequencies from human data.")
    parser.add_argument("directories", nargs='+', help="List of directories containing seed files.")
    parser.add_argument("--specific_epoch", type=int, default=29, help="Specify which epoch to analyze (default: 29).")
    args = parser.parse_args()
    # seeds_directories = ['../condition1_human/dump_context/msg_spk_seed111']
    freq_dist_agent(args.directories, args.specific_epoch, plot_all=False) 