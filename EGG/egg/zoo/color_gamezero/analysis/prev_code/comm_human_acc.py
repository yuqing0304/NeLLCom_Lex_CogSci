import csv
import ast

# Function to calculate accuracy
def calculate_accuracy(results, reference_file):
    correct_count = 0
    total_count = len(results)
    
    # Load reference data
    reference_data = []
    with open(reference_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Parse arrays and other fields from reference file
            arrays = [ast.literal_eval(entry) for entry in row[:3]]
            reordered_arrays = [ast.literal_eval(entry) for entry in row[3:6]]
            label = int(row[6])
            category = row[7]
            reference_data.append((arrays, reordered_arrays, label, category))
    
    # Compare results with reference data
    for result in results:
        arrays, category, reordered_arrays, label, _ = result
        for ref_arrays, ref_reordered_arrays, ref_label, ref_category in reference_data:
            if arrays == ref_arrays and reordered_arrays == ref_reordered_arrays and category == ref_category:
                correct_count += 1
                break
    
    # Calculate accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0
    return accuracy

# Function to read results from output.txt file
def read_results_from_file(file_path):
    results = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                # Remove leading/trailing whitespace and handle the structure of the line
                line = line.strip()

                # Split the parts manually to handle the format
                parts = line.split(" -> ")

                # First part is the arrays part
                arrays_part = parts[0]
                arrays = ast.literal_eval(arrays_part)

                # Second part is the category
                category = parts[1].strip('"')

                # Third part is the reordered arrays
                reordered_arrays_part = parts[2]
                reordered_arrays = ast.literal_eval(reordered_arrays_part)

                # Last part is the label and extra value. We split by the last space to handle it properly
                label_and_extra = parts[3].split(" ")
                label = int(label_and_extra[0])
                extra = int(label_and_extra[1]) if len(label_and_extra) > 1 else 0

                # Append the tuple
                results.append((arrays, category, reordered_arrays, label, extra))

            except Exception as e:
                print(f"Error reading line: {line}. Exception: {e}")
    return results

# Function to calculate and save average accuracy over all seeds for each epoch
def calculate_and_save_average_accuracy(reference_file, seed_list, num_epochs=20, output_file='average_accuracy.txt'):
    with open(output_file, 'w') as f:
        # Process "Initial Eval" separately
        initial_eval_accuracies = []
        for seed in seed_list:
            initial_eval_file = f'/projects/0/prjs1171/slearning/EGG/egg/zoo/color_gamezero/resultallhuman/dump_exp/msg_rf_seed{seed}/output_Initial Eval.txt'
            results = read_results_from_file(initial_eval_file)
            accuracy = calculate_accuracy(results, reference_file)
            initial_eval_accuracies.append(accuracy)

        # Calculate the average accuracy for "Initial Eval"
        initial_eval_avg = sum(initial_eval_accuracies) / len(initial_eval_accuracies)
        f.write(f"Initial Eval: Average Accuracy = {initial_eval_avg:.2f}\n")
        print(f"Initial Eval: Average Accuracy = {initial_eval_avg:.2f}")

        # Then, process regular epochs from 0 to 19
        for epoch in range(num_epochs):
            total_accuracy = 0
            num_seeds = len(seed_list)

            # Loop through each seed and calculate the accuracy for the current epoch
            for seed in seed_list:
                # Construct the filename for the current epoch and seed
                results_file = f'/projects/0/prjs1171/slearning/EGG/egg/zoo/color_gamezero/resultallhuman/dump_exp/msg_rf_seed{seed}/output_{epoch}.txt'
                
                # Read results from the file
                results = read_results_from_file(results_file)

                # Calculate accuracy for this seed and epoch
                accuracy = calculate_accuracy(results, reference_file)
                total_accuracy += accuracy

            # Calculate the average accuracy for the current epoch over all seeds
            average_accuracy = total_accuracy / num_seeds
            
            # Write the average accuracy for this epoch to the file
            f.write(f"Epoch {epoch}: Average Accuracy = {average_accuracy:.2f}\n")
            print(f"Epoch {epoch}: Average Accuracy = {average_accuracy:.2f}")

# List of seeds
# seed_list = [100, 111, 222, 333, 444, 555, 666, 777, 888, 999]
seed_list = [111]

# Reference file containing the correct answers
reference_file = "new_output.csv"

# Calculate and save average accuracy for each epoch and "Initial Eval" over all seeds
calculate_and_save_average_accuracy(reference_file, seed_list)
