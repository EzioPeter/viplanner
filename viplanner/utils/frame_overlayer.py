import os

import cv2


def load_img_filenames_from_directory(directory):
    images = []
    img_filenames = os.listdir(directory)
    img_filenames.sort()
    for filename in img_filenames:
        if not filename.endswith(".png"):
            continue
        images.append(filename)
    return images


def make_black_pixels_transparent(image):
    # Convert the image to RGBA format
    rgba_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # Create a mask for black pixels
    black_mask = (rgba_image[:, :, 0] == 0) & (rgba_image[:, :, 1] == 0) & (rgba_image[:, :, 2] == 0)

    # Set the alpha channel to 0 for black pixels
    rgba_image[black_mask] = [0, 0, 0, 0]

    return rgba_image


def overlay_images(image1, image2):
    # Ensure both images have the same dimensions
    if image1.shape[:2] != image2.shape[:2]:
        raise ValueError("Images must have the same dimensions for overlaying.")

    # Overlay the images
    overlay = cv2.addWeighted(image1, 1, cv2.cvtColor(image2, cv2.COLOR_BGR2BGRA), 1, 0)

    return overlay


def merge_images(image1, image2):
    # Create a mask for black pixels
    black_mask = (image1[:, :, 0] <= 5) & (image1[:, :, 1] <= 5) & (image1[:, :, 2] <= 5)

    image2[~black_mask] = image1[~black_mask]
    return image2


def main():
    # Input directories  (dir1 = dark images, dir2 = normal images)
    dir1 = "/home/pascal/viplanner/imperative_learning/models/plannernet_env2azQ1b91cZZ_cam_mount_ep100_inputDep_costSem_optimSGD_new_cam_mount_combi_lossWidthMod_wgoal4.0_warehouse_depth/eval_warehouse_multiple_shelves_without_ppl/render_video/warehouse_dark_frames"
    dir2 = "/home/pascal/viplanner/imperative_learning/models/plannernet_env2azQ1b91cZZ_cam_mount_ep100_inputDepSem_costSem_optimSGD_new_cam_mount_combi_lossWidthMod_wgoal4.0_warehouse/eval_warehouse_multiple_shelves_without_ppl/render_video/warehouse_frames"
    # Output directory
    output_dir = "/home/pascal/viplanner/imperative_learning/models/plannernet_env2azQ1b91cZZ_cam_mount_ep100_inputDepSem_costSem_optimSGD_new_cam_mount_combi_lossWidthMod_wgoal4.0_warehouse/eval_warehouse_multiple_shelves_without_ppl/render_video/overlay_frames"

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load images from both directories
    imagefilename1 = load_img_filenames_from_directory(dir1)
    imagefilename2 = load_img_filenames_from_directory(dir2)

    # Ensure both directories have the same number of images
    if len(imagefilename1) != len(imagefilename2):
        print("Both directories do contain the same number of images.")
        if len(imagefilename1) < len(imagefilename2):
            imagefilename1 = imagefilename1 + [imagefilename1[-1]] * (len(imagefilename2) - len(imagefilename1))
        else:
            imagefilename2 = imagefilename2 + [imagefilename2[-1]] * (len(imagefilename1) - len(imagefilename2))

    for idx, curr_img1_file in enumerate(imagefilename1):
        image1 = cv2.imread(os.path.join(dir1, curr_img1_file))
        image2 = cv2.imread(os.path.join(dir2, imagefilename2[idx]))

        # Make black pixels transparent in the first image
        # transparent_image1 = make_black_pixels_transparent(image1)

        # # Overlay both images
        # overlaid_image = overlay_images(transparent_image1, image2)

        overlaid_image = merge_images(image1, image2)

        # Save the result to the output directory
        output_path = os.path.join(output_dir, "result_" + f"{idx}".zfill(4) + ".png")
        assert cv2.imwrite(output_path, overlaid_image)
        print(f"Saved {idx}")


if __name__ == "__main__":
    main()
