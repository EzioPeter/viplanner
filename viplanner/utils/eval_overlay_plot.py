import cv2
import numpy as np

def extract_frames(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Unable to open video file: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_gap = total_frames // num_frames

    ret, initial_frame = cap.read()
    if not ret:
        print(f"Can't receive frame (stream end?). Exiting ...")
        return None

    output = initial_frame.astype('float32') / 255
    frame_id = 0
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_gap == 0:
            diff = np.abs(initial_frame.astype('int64') - frame.astype('int64'))
            changed_mask = np.any(diff > 20, axis=-1)  # mask for changed pixels
            # Alpha blending: source image * alpha + background image * (1 - alpha)
            output = frame.astype('float32') / 255 * changed_mask[:, :, np.newaxis] * 0.6 + output * changed_mask[:, :, np.newaxis] * 0.4 + output * (1 - changed_mask[:, :, np.newaxis])
            frame_count += 1
        frame_id += 1
        if frame_count >= num_frames:
            break

    output = (output*255).astype('uint8')

    cap.release()
    cv2.destroyAllWindows()
    return output

def save_image(image, path):
    cv2.imwrite(path, image)

if __name__ == "__main__":
    video_path = '/home/pascal/Downloads/microsoft_demo_cut.mp4' # specify your video path here
    output_path = '/home/pascal/Downloads/output.png' # specify your output image path here
    num_frames = 15 # specify the number of frames you want to extract and overlay
    overlayed_frame = extract_frames(video_path, num_frames)
    if overlayed_frame is not None:
        save_image(overlayed_frame, output_path)
