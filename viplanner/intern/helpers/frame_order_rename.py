import os


def rename_files_with_numbering(directory):
    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    # Get a list of files in the directory
    files = os.listdir(directory)
    files.sort()

    # Initialize a counter for the numbering
    count = 0

    # Iterate through the files and rename them
    for filename in files:
        if not filename.endswith(".png"):
            continue

        # Construct the new file name
        new_name = f"{count:04d}_rgb.png"  # Use leading zeros for consistent naming
        new_path = os.path.join(directory, new_name)

        # Construct the current file's path
        current_path = os.path.join(directory, filename)

        # Rename the file
        os.rename(current_path, new_path)

        # Increment the counter
        count += 1


if __name__ == "__main__":
    target_directory = "/home/pascal/Downloads/SRD_data/video_sem_after_projected"
    # target_directory = "/media/pascal/NavigationData/PascalRothData/bag/2023_09_07_crosswalk_sidewalk_wet_success/video_sem_after_projected"
    rename_files_with_numbering(target_directory)
