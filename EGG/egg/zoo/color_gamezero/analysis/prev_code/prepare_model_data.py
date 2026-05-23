import ast
import pandas as pd
import pickle
import argparse
from collections import Counter
from utils import rgb_to_cielab, compute_cielab_distance

# Argument parsing for command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('data_csv', help="csv file to analyze")
parser.add_argument('informativeness_dict', help="informativeness dictionary (.pkl file)")
parser.add_argument('prototypes_dict', help="prototypes dictionary (.pkl file)")
parser.add_argument('agent_model_data1', help="csv file for agent model data 1")
parser.add_argument('agent_model_data2', help="csv file for agent model data 2")
args = parser.parse_args()

# Process the input file (e.g., my_data.csv)
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

            if outcome_1 != outcome_2:
                continue  # Skip unsuccessful trials

            try:
                rgb_tuple = ast.literal_eval(rgb_str)  # Convert string "[R, G, B]" to a list
                if isinstance(rgb_tuple, (list, tuple)) and len(rgb_tuple) == 3:
                    data.append((color_name, tuple(rgb_tuple)))  # Store as a tuple
            except (SyntaxError, ValueError):
                continue  # Skip malformed RGB entries

    return data

# Load the data
data = process_myfile(args.data_csv)
informativeness = pickle.load(open(args.informativeness_dict, "rb"))
prototypes = pickle.load(open(args.prototypes_dict, "rb"))

# Process the data for agent_model_data1 (repeated chips as target)
targets = Counter([item[0] for item in data])

# Model 1A: Filter and organize data for repeated targets
color_names = []
for color, count in targets.items():
    if count >= 2:  # Only include colors that appear at least twice
        color_data = [item for item in data if item[0] == color]
        rgb_values = [item[1] for item in color_data]
        
        # Calculate other required variables for model 1A
        hard_dist = min(rgb_values)  # This should be a proper calculation (e.g., RGB distance or something relevant)
        easy_dist = max(rgb_values)  # Adjust as needed for your model logic
        condition = "some_condition"  # Placeholder - Replace with actual data if available
        outcome = "some_outcome"  # Placeholder - Replace with actual outcome if available

        color_names.append([color, rgb_values, hard_dist, easy_dist, condition, outcome])

# Organize into dictionary with necessary data
dic = {}
for item in color_names:
    color = item[0]
    rgb_values = item[1]
    hard_dist = item[2]
    easy_dist = item[3]
    condition = item[4]
    outcome = item[5]
    dic[color] = list(zip(rgb_values, [hard_dist]*len(rgb_values), [easy_dist]*len(rgb_values), [condition]*len(rgb_values), [outcome]*len(rgb_values)))

# Organize final result
result = []
for key, values in dic.items():
    for value in values:
        rgb_value, hard_dist, easy_dist, condition, outcome = value
        result.append((key, rgb_value, informativeness.get(key, None), hard_dist, easy_dist, condition, outcome))

df1 = pd.DataFrame(result, columns=["color_name", "rgb_value", "name_informativeness", "hard_distance", "easy_distance", "condition", "outcome"])

# For typicality (calculating CIELAB distance from prototypes)
tar_atyp = []
for name, rgb_value in zip(df1['color_name'], df1['rgb_value']):
    rgb_lab = rgb_to_cielab(rgb_value[0], rgb_value[1], rgb_value[2])
    tar_atyp.append(compute_cielab_distance(rgb_lab, prototypes.get(name, None)))

df1['typicality'] = tar_atyp

# Save the processed data for model 1A (repeated chips as target)
df1.to_csv(args.agent_model_data1)

# Model 1B: Use all chips for prediction (no repetition condition)
color_names = []
for color, count in targets.items():
    color_data = [item for item in data if item[0] == color]
    rgb_values = [item[1] for item in color_data]
    
    # Calculate other required variables for model 1B
    hard_dist = min(rgb_values)
    easy_dist = max(rgb_values)
    condition = "some_condition"
    outcome = "some_outcome"

    color_names.append([color, rgb_values, hard_dist, easy_dist, condition, outcome])

# Organize into dictionary with necessary data
dic = {}
for item in color_names:
    color = item[0]
    rgb_values = item[1]
    hard_dist = item[2]
    easy_dist = item[3]
    condition = item[4]
    outcome = item[5]
    dic[color] = list(zip(rgb_values, [hard_dist]*len(rgb_values), [easy_dist]*len(rgb_values), [condition]*len(rgb_values), [outcome]*len(rgb_values)))

# Organize final result for model 1B
result = []
for key, values in dic.items():
    for value in values:
        rgb_value, hard_dist, easy_dist, condition, outcome = value
        result.append((key, rgb_value, informativeness.get(key, None), hard_dist, easy_dist, condition, outcome))

df2 = pd.DataFrame(result, columns=["color_name", "rgb_value", "name_informativeness", "hard_distance", "easy_distance", "condition", "outcome"])

# For typicality (calculating CIELAB distance from prototypes)
tar_atyp = []
for name, rgb_value in zip(df2['color_name'], df2['rgb_value']):
    rgb_lab = rgb_to_cielab(rgb_value[0], rgb_value[1], rgb_value[2])
    tar_atyp.append(compute_cielab_distance(rgb_lab, prototypes.get(name, None)))

df2['typicality'] = tar_atyp

# Save the processed data for model 1B (all chips)
df2.to_csv(args.agent_model_data2)
