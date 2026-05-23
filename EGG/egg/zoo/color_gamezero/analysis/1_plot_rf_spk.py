import os
import re
import numpy as np
import matplotlib.pyplot as plt

def process_log_files(log_folders, log_prefix):
    """
    Reads multiple log folders (each corresponding to different seeds),
    extracts accuracy per epoch, and returns mean and std deviation across seeds.
    
    Parameters:
        log_folders (list of str): List of seed-specific log folders for a condition.
        log_prefix (str): Prefix for log files.
    
    Returns:
        unique_epochs (ndarray): Sorted unique epochs across all seeds.
        mean_accuracies (ndarray): Mean accuracy per epoch.
        std_accuracies (ndarray): Standard deviation per epoch.
    """
    all_epochs = []
    all_accuracies = []

    for log_folder in log_folders:
        print(f"log_folder, {log_folder}")
        rename_files(log_folder)

    for log_folder in log_folders:
        log_files = [f for f in os.listdir(log_folder) if f.startswith(log_prefix) and f.endswith('.txt')]

        print(f"Reading from {log_folder}: {log_files}")

        epoch_accuracy = {}

        for log_file in log_files:

            match = re.search(r'output_epoch(\d+).txt', log_file)
            if match:
                epoch = int(match.group(1))
            else:
                continue

            correct = 0
            total = 0
            log_path = os.path.join(log_folder, log_file)

            with open(log_path, 'r') as f:
                for line in f:
                    match = re.search(r'->\s*(\w+)\s*\((\w+)\)\s*->', line)
                    if match:
                        color1, color2 = match.groups()
                        if color1 == color2:
                            correct += 1
                        total += 1

            accuracy = correct / total if total > 0 else 0
            epoch_accuracy[epoch] = accuracy

        if epoch_accuracy:
            all_epochs.append(np.array(sorted(epoch_accuracy.keys())))
            all_accuracies.append(np.array([epoch_accuracy[e] for e in sorted(epoch_accuracy.keys())]))

    if not all_epochs:
        print("No valid data found.")
        return None, None, None

    # Align epochs across seeds
    unique_epochs = np.unique(np.concatenate(all_epochs))
    
    # Interpolate missing values for consistency
    all_interpolated_accuracies = []
    for epochs, accuracies in zip(all_epochs, all_accuracies):
        interpolated_accuracies = np.interp(unique_epochs, epochs, accuracies)
        all_interpolated_accuracies.append(interpolated_accuracies)

    all_interpolated_accuracies = np.array(all_interpolated_accuracies)

    # Compute mean and std
    mean_accuracies = np.mean(all_interpolated_accuracies, axis=0)
    std_accuracies = np.std(all_interpolated_accuracies, axis=0)

    return unique_epochs, mean_accuracies, std_accuracies



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



def plot_accuracy(conditions, output_file):
    """
    Plots accuracy over epochs for multiple conditions.

    Parameters:
        conditions (dict): Dictionary with keys as condition labels and values as lists of seed folders.
        output_file (str): Path to save the plot.
    """
    plt.figure(figsize=(8, 5))

    for label, log_folders in conditions.items():
        epochs, mean_accuracies, std_accuracies = process_log_files(log_folders, "output_")
        if epochs is not None:
            plt.plot(epochs, mean_accuracies, label=label)
            plt.fill_between(epochs, mean_accuracies - std_accuracies, mean_accuracies + std_accuracies, alpha=0.3)

    # plt.figure(figsize=(10, 6))  # Set figure size (width=10, height=6 in inches)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Speaking accuracy (RL1)")
    plt.xticks(np.arange(0, 31, 5))
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.show()

# conditions = {
#     "SL w/o context (zeroed) + RL w/o context": ["../condition3_generated/dump/msg_rf_seed111", "../condition3_generated/dump/msg_rf_seed222"],
#     "SL w/ context + RL w/ context": ["../condition3_generated/dump_context/msg_rf_seed111", "../condition3_generated/dump_context/msg_rf_seed222"],
#     "SL w/o context (zeroed) + RL w/ context": ["../condition3_generated/dump_exp/msg_rf_seed111", "../condition3_generated/dump_exp/msg_rf_seed222"]
# }

# conditions = {
#     "SL w/o context (zeroed) + RL w/o context": ["../condition1_human/dump/msg_rf_seed111", "../condition1_human/dump/msg_rf_seed222"],
#     "SL w/ context + RL w/ context": ["../condition1_human/dump_context/msg_rf_seed111", "../condition1_human/dump_context/msg_rf_seed222"],
#     "SL w/o context (zeroed) + RL w/ context": ["../condition1_human/dump_exp/msg_rf_seed111", "../condition1_human/dump_exp/msg_rf_seed222"]
# }

conditions = {
    "SL w/o context (zeroed) + RL w/o context": ["../condition3_generated/experiment_human/dump_context/msg_rf_seed111", "../condition3_generated/experiment_human/dump_context/msg_rf_seed222"],
    "SL w/ context + RL w/ context": ["../condition3_generated/experiment_human/dump_context/msg_rf_seed111", "../condition3_generated/experiment_human/dump_context/msg_rf_seed222"],
    "SL w/o context (zeroed) + RL w/ context": ["../condition3_generated/experiment_human/dump_context/msg_rf_seed111", "../condition3_generated/experiment_human/dump_context/msg_rf_seed222"]
}

output_file = "acc_spk_rf_generated.png"
plot_accuracy(conditions, output_file)
