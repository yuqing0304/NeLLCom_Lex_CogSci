import os
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import re
import argparse

### note: # ======= not filter by communication success ======= 
### python 3_entropy_rl_human.py ../condition1_human/dump_context/msg_rf_seed111
### python 3_entropy_rl_human.py ../condition1_human/dump_context/msg_rf_seed222


def plot_entropy_over_epochs(seeds_directories):
    for dir in seeds_directories:
        rename_files(dir)
        if "spk" in dir:
            if "human" in dir:
                plot_type = "spk_human"
            elif "generated" in dir:
                plot_type = "spk_generated"
        elif "rf" in dir:
            if "human" in dir:
                plot_type = "rf_human"
            elif "generated" in dir:
                plot_type = "rf_generated"
    """Plot entropy as a function of epoch with a minimal style."""
    epoch_color_names, label_color_counts = aggregate_color_name_counts(seeds_directories)
    epochs = sorted(epoch_color_names.keys(), key=lambda x: int(x))  # Sort epochs numerically
    entropies = [calculate_entropy_based_on_word_frequency(epoch_color_names[epoch]) for epoch in epochs]

    print(f"label_color_counts, {label_color_counts}")

    # Compute entropy for label colors
    label_color_entropy = calculate_entropy_based_on_word_frequency(label_color_counts)

    # Create directory if it doesn't exist
    output_dir = "./entropy"
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, entropies, marker='o', linestyle='-', color='black', markersize=5, linewidth=1)

    plt.axhline(y=label_color_entropy, color='red', linestyle='--', linewidth=1, label='Entropy of human production')

    # Add text annotation near the line
    plt.text(
        x=30 * 0.95,  # Adjust based on your x-axis range
        y=label_color_entropy + 0.02,  # Slightly above the line
        s=f"{label_color_entropy:.4f}",
        color='red',
        fontsize=10,
        verticalalignment='bottom'
    )

    # Minimalist styling
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Entropy', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title('Entropy Over Epochs', fontsize=14)
    plt.ylim(0, 4)  # Ensure y-axis starts above 0
    plt.gca().spines[['top', 'right']].set_visible(False)  # Remove top/right borders
    plt.gca().spines[['left', 'bottom']].set_linewidth(0.8)  # Thin axis lines
    # plt.show()
    # Save the plot
    output_path = os.path.join(output_dir, f"entropy_{plot_type}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Plot saved to {output_path}")



def aggregate_color_name_counts(seeds_directories):
    """Aggregate color name counts dynamically from the available epoch files"""
    epoch_color_names = {}
    label_color_counts = Counter()  # Store label color counts separately

    for directory in seeds_directories:
        print(f"Processing seed directory: {directory}")

        if "spk" in directory:
            process_function = process_file_freq_spk
            files = [f for f in os.listdir(directory) if f.startswith('sender_epoch') and f.endswith('.txt')]
        elif "rf" in directory:
            process_function = process_file_freq
            files = [f for f in os.listdir(directory) if f.startswith('output_epoch') and f.endswith('.txt')]
        else:
            print(f"Warning: Unknown directory type in {directory}. Skipping...")
            continue  

        for file in files:
            epoch = extract_epoch_from_filename(file)
            if epoch:
                data = process_function(os.path.join(directory, file))
                color_names = accumulate_color_names(data)

                if epoch not in epoch_color_names:
                    epoch_color_names[epoch] = Counter()
                epoch_color_names[epoch].update(color_names)

                # # Also collect label colors if processing "spk"
                # if "spk" in directory:
                #     label_colors = accumulate_label_colors(data)  # Extract label colors
                #     label_color_counts.update(label_colors)

        # Extract label colors **only from the last epoch** for "spk"
        if "rf" in directory:
            last_epoch_file = files[-1]  # Get the latest epoch file
            data = process_function(os.path.join(directory, last_epoch_file))
            label_colors = accumulate_label_colors(data)  # Extract label colors only from the last epoch
            label_color_counts = Counter(label_colors)  # Replace instead of updating

    return epoch_color_names, label_color_counts  # Return both normal and label color counts


def accumulate_label_colors(data):
    """Extract label colors from the processed spk data"""
    label_colors = []
    for entry in data:
        if len(entry) > 2:  # Ensure label color exists in the entry
            label_colors.append(entry[2])  # Extract label color
    print(f"label_colors, {label_colors}")
    return label_colors


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



def process_file_freq(file_path):
    """Process the file and extract color names, skipping unsuccessful trials."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split("->")
            if len(parts) < 5:  
                continue  # Skip if the line doesn't have enough parts

            hls_str = parts[0].strip()
            color_part = parts[1].strip()


            # Extract main color and alternative color (if present in parentheses)
            match = re.match(r"(\w+)\s*\((.*?)\)", color_part)

            if match:
                color_name = match.group(1)  # First part before parentheses
                label_colors = match.group(2)   # Inside parentheses
            else:
                color_name = color_part 

            outcome_1 = parts[3].strip()
            outcome_2 = parts[4].strip()

            # ======= filter by communication success ======= 
            # if outcome_1 != outcome_2:
            #     continue  # Skip unsuccessful trials
            data.append([hls_str, color_name, label_colors])

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

            # ======= filter by communication success ======= 
            # if color_name != label_color:
            #     continue  # Skip if label color doesn't match extracted color name

            data.append([hls_str, color_name])
    # print(f"data length, {len(data)}")

    return data


def accumulate_color_names(data):
    """Accumulate color names from the data, removing any bracketed content"""
    color_names = []
    for entry in data:
        color_name = entry[1]
        cleaned_color_name = re.sub(r"\s*\(.*?\)", "", color_name)  # Remove content inside parentheses
        color_names.append(cleaned_color_name)
        # print(f"color_names, {color_names}")
    return color_names



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




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot entropy over epochs and save it.")
    parser.add_argument("seeds_directories", nargs="+", help="List of seed directories to process.")
    args = parser.parse_args()

    # Process each seed directory one by one
    plot_entropy_over_epochs(args.seeds_directories)  # Pass the entire list



