import os
import shutil

def copy_images_with_prefix(source_dirs, dest_dir, file_name):
    # global_counter
    counter = 0
    # Create the destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    for curr_dir in source_dirs:
        files = os.listdir(curr_dir)
        files.sort()
        # Iterate over all files in the source directory
        for filename in files:
            # Check if the file is an image (you can modify the condition as per your specific image file extensions)
            if filename.lower().endswith('.png') and filename.startswith(file_name):
                # Construct the new filename by adding the source directory name as a prefix
                new_filename = f"{file_name}_" + f"{counter}".zfill(4) + ".png"

                # Copy the image file from the source directory to the destination directory
                shutil.copy2(os.path.join(curr_dir, filename), os.path.join(dest_dir, new_filename))

                counter += 1

# Example usage
source_dirs = [
    "/home/pascal/viplanner/imperative_learning/code/iPlanner/iplanner/models/eval_Town01_Opt_paper/viplanner_rgb/crosswalk_paper_changed_waypoint0_of_4",
    "/home/pascal/viplanner/imperative_learning/code/iPlanner/iplanner/models/eval_Town01_Opt_paper/viplanner_rgb/crosswalk_paper_changed_waypoint1_of_4",
    "/home/pascal/viplanner/imperative_learning/code/iPlanner/iplanner/models/eval_Town01_Opt_paper/viplanner_rgb/crosswalk_paper_changed_waypoint2_of_4",
    "/home/pascal/viplanner/imperative_learning/code/iPlanner/iplanner/models/eval_Town01_Opt_paper/viplanner_rgb/crosswalk_paper_changed_waypoint3_of_4",
]
dest_dir = "/home/pascal/viplanner/imperative_learning/code/iPlanner/iplanner/models/eval_Town01_Opt_paper/viplanner_rgb/crosswalk_paper_changed_waypoint_all_sem"
file_name = "semantic_segmentation_step"

copy_images_with_prefix(source_dirs, dest_dir, file_name)