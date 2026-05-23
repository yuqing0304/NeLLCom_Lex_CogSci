import pandas as pd
import os
import glob

# List of base directories to process
base_dirs = ["./dump_context"]

# Define the seed list you want to include
# seed_list = [111, 222, 333, 444, 555, 666, 777, 888, 999]
# seed_list = [111, 123, 333, 345, 444, 555, 567, 666, 777, 912]
seed_list = [111, 123, 333, 345, 444, 555, 567, 666, 777, 912]


for base_dir in base_dirs:
    # Find all epoch0_model_data.csv files in seed subfolders
    csv_files = glob.glob(os.path.join(base_dir, "msg_rf_seed*/epoch30_model_data.csv"))

    all_dfs = []
    for file_path in csv_files:
        # Extract the seed number from the folder name
        seed_folder = os.path.basename(os.path.dirname(file_path))
        seed_number = int(seed_folder.replace("msg_rf_seed", ""))  # make it int for comparison

        if seed_number in seed_list:  # only include selected seeds
            df = pd.read_csv(file_path)
            df["seed"] = seed_number
            all_dfs.append(df)

    if all_dfs:  # only concatenate if we actually got some
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Create output filename based on base_dir name
        base_name = os.path.basename(base_dir.rstrip('/'))
        output_filename = f"{base_name}_epoch30_model_data.csv"
        combined_df.to_csv(output_filename, index=False)

        print(f"Combined CSV saved as {output_filename}")
    else:
        print(f"No matching seeds found in {base_dir}")
