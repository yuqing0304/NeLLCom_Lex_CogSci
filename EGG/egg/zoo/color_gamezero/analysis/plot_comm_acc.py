import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

def read_and_plot_accuracy(log_folder, log_prefix, label, accuracy_type):
    """
    Reads JSON log files with the specified prefix from a folder, extracts the specified accuracy metrics,
    calculates the average and standard deviation over all seeds, and plots it as a function of epoch.
    """
    # Collect all log files matching the prefix
    log_files = [f for f in os.listdir(log_folder) if f.startswith(log_prefix) and f.endswith('.txt')]
    print(f"Reading from {log_folder}: {log_files}")
    
    all_epochs = []
    all_accs = []

    for log_file in log_files:
        log_path = os.path.join(log_folder, log_file)
        epochs = []
        accs = []

        # Read the file and extract epoch and selected accuracy type
        with open(log_path, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line)
                    if log_entry.get("mode") == "test":  # Filter for test mode
                        epoch = log_entry["epoch"]
                        acc = log_entry[accuracy_type]  # Use the selected accuracy type
                        
                        epochs.append(epoch)
                        accs.append(acc)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line in file: {log_file}")

        if epochs and accs:
            epochs = np.array(epochs, dtype=int)
            accs = np.array(accs, dtype=float)
            all_epochs.append(epochs)
            all_accs.append(accs)
    
    if not all_epochs or not all_accs:
        print(f"No valid data found in {log_folder} with prefix {log_prefix}.")
        return

    all_epochs = np.concatenate(all_epochs)
    all_accs = np.concatenate(all_accs)

    # Unique epochs
    unique_epochs = np.unique(all_epochs)

    # Function to calculate average and standard deviation
    def calculate_avg_std(accs):
        avg_accs = []
        std_accs = []
        for epoch in unique_epochs:
            epoch_indices = all_epochs == epoch
            avg_acc = np.mean(accs[epoch_indices])
            std_acc = np.std(accs[epoch_indices])
            avg_accs.append(avg_acc)
            std_accs.append(std_acc)
        return np.array(avg_accs), np.array(std_accs)

    # Calculate average and std for the selected accuracy type
    avg_accs, std_accs = calculate_avg_std(all_accs)

    # Plot the chosen accuracy with shaded regions for standard deviation
    plt.plot(unique_epochs, avg_accs, label=label)
    plt.fill_between(unique_epochs, avg_accs - std_accs, avg_accs + std_accs, alpha=0.3)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Plot accuracy from training logs.")
    parser.add_argument('--log_folder1', type=str, required=True, help="First folder containing log files")
    parser.add_argument('--log_prefix1', type=str, required=True, help="Prefix of the first log files")
    parser.add_argument('--log_folder2', type=str, required=True, help="Second folder containing log files")
    parser.add_argument('--log_prefix2', type=str, required=True, help="Prefix of the second log files")
    parser.add_argument('--log_folder3', type=str, required=True, help="Third folder containing log files")
    parser.add_argument('--log_prefix3', type=str, required=True, help="Prefix of the third log files")
    parser.add_argument('--log_folder4', type=str, required=True, help="Fourth folder containing log files")  # New fourth condition
    parser.add_argument('--log_prefix4', type=str, required=True, help="Prefix of the fourth log files")  # New fourth condition
    parser.add_argument('--log_folder5', type=str, required=True, help="Fifth folder containing log files")  # New fifth condition
    parser.add_argument('--log_prefix5', type=str, required=True, help="Prefix of the fifth log files")  # New fifth condition
    parser.add_argument('--accuracy_type', type=str, choices=['acc', 'acc_far', 'acc_close', 'acc_split'], required=True, help="Accuracy type to plot (acc, acc_far, acc_close, acc_split)")
    parser.add_argument('--output_plot', type=str, required=True, help="Output plot file name")
    
    # Parse the arguments
    args = parser.parse_args()

    # Plot for each condition
    read_and_plot_accuracy(args.log_folder1, args.log_prefix1, "far:close 0/100", args.accuracy_type)
    read_and_plot_accuracy(args.log_folder2, args.log_prefix2, "far:close 33/67", args.accuracy_type)
    read_and_plot_accuracy(args.log_folder3, args.log_prefix3, "far:close 50/50", args.accuracy_type)
    read_and_plot_accuracy(args.log_folder4, args.log_prefix4, "far:close 67/33", args.accuracy_type)  # New condition 4
    read_and_plot_accuracy(args.log_folder5, args.log_prefix5, "far:close 100/0", args.accuracy_type)  # New condition 5

    # Add labels, legend, and title to the plot
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs. Epoch for {args.accuracy_type.capitalize()} Accuracy")
    plt.legend(loc='lower right')
    plt.grid(True)

    # Set y-axis to range from 0 to 1
    plt.ylim(0.7, 1)
    
    # Save the plot to a file
    plt.savefig(args.output_plot, bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()



# python plot_comm_acc.py --log_folder1 ../condition3_generated/condition_slrl3/training_log_context --log_prefix1 log_rf --log_folder2 ../condition3_generated/condition_slrl4/training_log_context --log_prefix2 log_rf --log_folder3 ../condition3_generated/condition_slrl5/training_log_context --log_prefix3 log_rf --log_folder4 ../condition3_generated/condition_slrl6/training_log_context --log_prefix4 log_rf --log_folder5 ../condition3_generated/condition_slrl7/training_log_context --log_prefix5 log_rf --accuracy_type acc --output_plot accuracy_plot.png

# python plot_comm_acc.py --log_folder1 ../condition3_generated/experiment2/condition_a/training_log_context --log_prefix1 log_rf --log_folder2 ../condition3_generated/experiment2/condition_b/training_log_context --log_prefix2 log_rf --log_folder3 ../condition3_generated/experiment2/condition_b/training_log_context --log_prefix3 log_rf --log_folder4 ../condition3_generated/experiment2/condition_b/training_log_context --log_prefix4 log_rf --log_folder5 ../condition3_generated/experiment2/condition_b/training_log_context --log_prefix5 log_rf --accuracy_type acc --output_plot accuracy_plot.png
