import os
import matplotlib.pyplot as plt
from collections import Counter

def accumulate_color_names(data):
    """Accumulate color names from the data"""
    color_names = []
    for entry in data:
        color_name = entry[1]
        color_names.append(color_name)
    return color_names

def extract_epoch_from_filename(file_name):
    """Extract the epoch number from the filename or return None if not found"""
    try:
        print(f"Extracting epoch from: {file_name}")  # Debugging print
        # Extract epoch number from filenames like sender_epoch0.txt, sender_epoch1.txt, etc.
        epoch_str = file_name.split('sender_epoch')[1].split('.txt')[0]
        print(f"Extracted epoch_str: {epoch_str}")  # Debugging print
        return epoch_str  # Return the epoch as a string, e.g., 'epoch0', 'epoch1', etc.
    except ValueError:
        return None  # Return None if epoch extraction fails

def plot_color_name_histogram(color_names, epoch):
    """Plot and save frequency histogram for color names"""
    color_name_counts = Counter(color_names)
    
    # Plot frequency histogram
    plt.figure(figsize=(10, 6))
    plt.bar(color_name_counts.keys(), color_name_counts.values(), color='skyblue')
    plt.xlabel('Color Name')
    plt.ylabel('Frequency')
    plt.title(f'Color Name Frequency Histogram for {epoch}')
    plt.xticks(rotation=80, fontsize=16)
    plt.tight_layout()

    # Save the histogram as a PNG file
    output_plot_file = f'epoch{epoch}_color_name_histogram.png'
    plt.savefig(output_plot_file)
    plt.close()

def process_file_human(file_path):
    """Process the file and extract color names"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split("->")
            hls_str = parts[0].strip()
            color_name = parts[1].strip()

            # Extract the part after 'label='
            if "label=" in color_name:
                color_name = color_name.split("label=")[-1].strip().strip(")")
            # We don't need to parse the HLS values for this task
            data.append([hls_str, color_name])
    
    # Debugging output
    print(f"Processed {len(data)} lines from {file_path}")
    
    return data


def aggregate_color_name_counts(seeds_directories):
    """Aggregate color name counts across different seeds for each epoch"""
    epoch_color_names = {str(i): Counter() for i in range(21)}  # Assuming epochs 0 to 20

    # Iterate over each seed directory
    for directory in seeds_directories:
        print(f"Processing seed directory: {directory}")
        
        # List of renamed files to process for each seed (epoch 0 to 20)
        files = [
            f'{directory}/sender_epoch0.txt', f'{directory}/sender_epoch1.txt',
            f'{directory}/sender_epoch2.txt', f'{directory}/sender_epoch3.txt',
            f'{directory}/sender_epoch4.txt', f'{directory}/sender_epoch5.txt',
            f'{directory}/sender_epoch6.txt', f'{directory}/sender_epoch7.txt',
            f'{directory}/sender_epoch8.txt', f'{directory}/sender_epoch9.txt',
            f'{directory}/sender_epoch10.txt', f'{directory}/sender_epoch11.txt',
            f'{directory}/sender_epoch12.txt', f'{directory}/sender_epoch13.txt',
            f'{directory}/sender_epoch14.txt', f'{directory}/sender_epoch15.txt',
            f'{directory}/sender_epoch16.txt', f'{directory}/sender_epoch17.txt',
            f'{directory}/sender_epoch18.txt', f'{directory}/sender_epoch19.txt'
        ]
        
        # Iterate over the files for this seed
        for file in files:
            epoch = extract_epoch_from_filename(file)
            if epoch:
                print(f"Processing file: {file}")
                data = process_file_human(file)
                color_names = accumulate_color_names(data)
                
                # Aggregate color name counts for this epoch across seeds
                epoch_color_names[epoch].update(color_names)
                
                print(f"Accumulated {len(color_names)} color names for {epoch} in {directory}")
            else:
                print(f"Warning: Could not extract epoch from file {file}")

    return epoch_color_names

def plot_epoch_histograms(epoch_color_names):
    """Plot and save frequency histograms for each epoch with a fixed x-axis order"""

    id_to_colors = {0: 'mustard', 1: 'cyan', 2: 'maroon', 3: 'lavander', 4: 'medium', 5: 'blood', 6: 'turquoise', 7: 'purple', 8: 'blue', 9: 'grapes', 10: 'caca', 11: 'teal', 12: 'sky', 13: 'grass', 14: 'red', 15: 'seafoam', 16: 'aqua', 17: 'clay', 18: 'barney', 19: 'green', 20: 'concrete', 21: 'pumpkin', 22: 'drab', 23: 'tan', 24: 'neon', 25: 'olive', 26: 'lavender', 27: 'fuchsia', 28: 'gray', 29: 'magenta', 30: 'grape', 31: 'dull', 32: 'peach', 33: 'mint', 34: 'mauve', 35: 'yellow', 36: 'sage', 37: 'brown', 38: 'pink', 39: 'beige', 40: 'gold', 41: 'orange', 42: 'salmon', 43: 'bright', 44: 'seagreen', 45: 'violet', 46: 'khaki', 47: 'rose', 48: 'lime'}   

    # Define the fixed order of color names
    fixed_color_order = list(id_to_colors.values())

    for epoch, color_counts in epoch_color_names.items():
        # Ensure all colors are present, even if their count is 0
        complete_counts = {color: color_counts.get(color, 0) for color in fixed_color_order}

        # Plot histogram
        plt.figure(figsize=(12, 6))
        plt.bar(complete_counts.keys(), complete_counts.values(), color='skyblue')
        plt.xlabel('Color Name')
        plt.ylabel('Frequency')
        plt.title(f'Color Name Frequency for Epoch {epoch}')
        plt.xticks(rotation=80, fontsize=12)
        plt.tight_layout()

        # Save the histogram as a PNG file
        output_plot_file = f'epoch{epoch}_color_freq_human.png'
        plt.savefig(output_plot_file)
        plt.close()
        print(f"Saved histogram for Epoch {epoch}")

def main():
    # List of directories for the different seeds (adjust paths as necessary)
    # seeds_directories = ['./dump_context/msg_spk_seed111', './dump_context/msg_spk_seed222']
    seeds_directories = ['./dump_context/msg_spk_seed222']

    # Aggregate color name counts across seeds for each epoch
    epoch_color_names = aggregate_color_name_counts(seeds_directories)
    
    # Plot and save the frequency histograms for each epoch
    plot_epoch_histograms(epoch_color_names)

if __name__ == "__main__":
    main()


