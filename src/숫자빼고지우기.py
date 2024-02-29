import os
import re
import shutil

def extract_numbers_from_filename(filename):
    # Extract numbers from the filename
    numbers_match = re.search(r'\d+', filename)
    if numbers_match:
        return numbers_match.group()
    else:
        return None

def process_files(input_directory, output_directory):
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Iterate through all files in the input directory
    for filename in os.listdir(input_directory):
        input_file_path = os.path.join(input_directory, filename)

        # Check if it's a file (not a subdirectory)
        if os.path.isfile(input_file_path):
            # Extract numbers from the filename
            numbers_only = extract_numbers_from_filename(filename)

            if numbers_only:
                # Create a new filename with only the numbers
                new_filename = f"{numbers_only}.mp4"

                # Construct the output file path
                output_file_path = os.path.join(output_directory, new_filename)

                # Copy the file to the output directory with the new name
                shutil.copy(input_file_path, output_file_path)

# Example usage
input_dir = 'C:\\Users\\AIA\\Desktop\\ai\\AI_mini_project\\resource\\video_standardized_320\\'
output_dir = 'C:\\Users\\AIA\\Desktop\\ai\\AI_mini_project\\resource\\video_standardized_320320\\'

process_files(input_dir, output_dir)
