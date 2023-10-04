import os

import cv2
import numpy as np


def extract_frames(video_path, num_frames, path, overlay: bool = True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Unable to open video file: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_gap = total_frames // num_frames

    ret, initial_frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        return None

    if not overlay:
        os.makedirs(path, exist_ok=True)
    output = initial_frame.astype("float32") / 255
    frame_id = 0
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_gap == 0:
            if overlay:
                diff = np.abs(initial_frame.astype("int64") - frame.astype("int64"))
                changed_mask = np.any(diff > 20, axis=-1)  # mask for changed pixels
                # Alpha blending: source image * alpha + background image * (1 - alpha)
                output = (
                    frame.astype("float32") / 255 * changed_mask[:, :, np.newaxis] * 0.6
                    + output * changed_mask[:, :, np.newaxis] * 0.4
                    + output * (1 - changed_mask[:, :, np.newaxis])
                )
            else:
                assert cv2.imwrite(f"{path}/frame_{frame_count}.png", frame)
            frame_count += 1
        frame_id += 1
        if frame_count >= num_frames:
            break

    output = (output * 255).astype("uint8")

    cap.release()
    cv2.destroyAllWindows()

    if overlay and output is not None:
        cv2.imwrite(path, output)
    return output


if __name__ == "__main__":
    video_path = "/home/pascal/Downloads/IMG_5569.MOV"  # specify your video path here
    output_path = "/home/pascal/Downloads/video_crosswalk_5569"  # output.png' # specify your output image path here
    num_frames = 200  # specify the number of frames you want to extract and overlay
    extract_frames(video_path, num_frames, output_path, False)
