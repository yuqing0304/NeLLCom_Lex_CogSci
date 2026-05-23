import re
import ast
import argparse

def extract_color_data(file_path, target_color):
    """
    Extract the first triplet of the first CIELAB list for all occurrences of a specific color.
    
    Args:
        file_path (str): Path to the input file.
        target_color (str): Color name to extract.
    
    Returns:
        list: All first triplets corresponding to the target color.
    """
    color_triplets = []

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split("->")
            if len(parts) < 2:
                continue

            # Extract color name
            color_part = parts[1].strip()
            match = re.match(r"(\w+)\s*\((.*?)\)", color_part)
            if not match:
                continue
            color_name = match.group(1)

            if color_name != target_color:
                continue

            # Extract first triplet
            try:
                cielab_lists = ast.literal_eval(parts[0].strip())
                first_triplet = cielab_lists[0]  # first triplet in the first list
                color_triplets.append((first_triplet, color_name))
            except (SyntaxError, ValueError, IndexError):
                continue

    return color_triplets


def main():
    parser = argparse.ArgumentParser(description="Extract first-triplet CIELAB values for a target color.")
    parser.add_argument("file_path", type=str, help="Path to the input file.")
    parser.add_argument("--color", type=str, required=True, help="Target color name to extract.")
    parser.add_argument("--output_file", type=str, default=None, help="Optional path to save extracted triplets.")
    args = parser.parse_args()

    data = extract_color_data(args.file_path, args.color)
    print(f"Found {len(data)} samples for color '{args.color}'")
    print(data[:5])  # preview first 5

    if args.output_file:
        with open(args.output_file, "w") as f:
            for triplet, color_name in data:
                f.write(f"{triplet}, {color_name}\n")
        print(f"Data saved to {args.output_file}")


if __name__ == "__main__":
    main()


# python narrow_low_words.py output_epoch0.txt --color barney --output_file barney_triplets.txt
