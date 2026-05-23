import pandas as pd
import random
import numpy as np
import ast
import os
import re
import colorsys
import colorspacious as cs
import matplotlib.pyplot as plt





# python word_informativeness.py --mode word_info --file_paths ../condition3_generated/dump_context/msg_rf_seed111/output_epoch30.txt --output_file_path ../condition3_generated/dump_context/epoch30.csv

# python word_informativeness.py --mode system --file_path ../condition3_generated/dump_context/msg_rf_seed111/output_epoch30.txt --word_info_path ../condition3_generated/dump_context/epoch30_informativeness.csv



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
            # color_name = parts[1].strip()
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
            condition = parts[5].strip()

            # # Skip unsuccessful trials
            # if outcome_1 != outcome_2:
            #     n += 1
            #     continue

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
    print(f"len(words_used), {len(words_used)}")
    
    # Compute mean informativeness
    lexical_system_informativeness = used_informativeness.mean()
    
    return lexical_system_informativeness


# ===========
def word_informativeness(file_paths, output_file_path):

    process_files(file_paths, output_file_path)

    # Process each file and compute informativeness independently
    # for file in files:
    print(f"Processing file: {output_file_path}")
    informativeness_df = process_file_info(output_file_path)

    # Print informativeness for the current file
    print(f"Informativeness for {output_file_path}:")
    print(informativeness_df)



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
            # color_name = parts[1].strip()  # Color name comes after '->'

            color_part = parts[1].strip()


            # Extract main color and alternative color (if present in parentheses)
            match = re.match(r"(\w+)\s*\((.*?)\)", color_part)

            if match:
                color_name = match.group(1)  # First part before parentheses
                label_colors = match.group(2)   # Inside parentheses
            else:
                color_name = color_part 


            outcome_1 = parts[3].strip() 
            # print(f"outcome_1, {outcome_1}")
            outcome_2 = parts[4].strip() 
            condition = parts[5].strip()

            # # Check if outcome is True
            # if outcome_1 != outcome_2:
            #     n = n+1
            #     continue  # Skip unsuccessful trials

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

    # # Plot informativeness distribution
    # plot_informativeness_distribution(informativeness_df, file_path)

    return informativeness_df




# def plot_informativeness_distribution(informativeness_df, file_path):
#     # Plot informativeness distribution for color names
#     plt.figure(figsize=(10, 6))
#     plt.bar(informativeness_df['color_name'], informativeness_df['informativeness'], color='skyblue')
#     plt.xlabel('Color Name')
#     plt.ylabel('Informativeness')
#     plt.title(f'Informativeness Distribution for {os.path.basename(file_path)}')
#     plt.xticks(rotation=80, fontsize=16)
#     plt.tight_layout()

#     # Save the plot as a PNG file
#     output_plot_file = file_path.replace('.txt', '_informativeness_distribution.png')
#     plt.savefig(output_plot_file)
#     plt.close()



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


# file_paths = ['../condition3_generated/dump_context/msg_rf_seed111/output_epoch30.txt']
# output_file_path = "../condition3_generated/dump_context/epoch30.txt"

# word_informativeness(file_paths, output_file_path)







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
            
            # if outcome_1 != outcome_2:
            #     continue  # Skip unsuccessful trials
            
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




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute word informativeness from files.")

    parser.add_argument("--file_paths", nargs="+", type=str, help="List of input text file paths for word informativeness computation.")
    parser.add_argument("--output_file_path", type=str, help="Output path for per-word informativeness CSV.")
    parser.add_argument("--file_path", type=str, help="Single input text file path for system-level informativeness.")
    parser.add_argument("--word_info_path", type=str, help="Path to the informativeness CSV file.")
    parser.add_argument("--mode", type=str, choices=["word_info", "system"], required=True, help="Mode: 'word_info' for per-word informativeness, 'system' for system-level informativeness.")

    args = parser.parse_args()

    if args.mode == "word_info":
        if not args.file_paths or not args.output_file_path:
            raise ValueError("You must provide --file_paths and --output_file_path for per-word informativeness.")
        word_informativeness(args.file_paths, args.output_file_path)
    
    elif args.mode == "system":
        if not args.file_path or not args.word_info_path:
            raise ValueError("You must provide --file_path and --word_info_path for system-level informativeness.")
        process_file_info_system(args.file_path, args.word_info_path)
