import cv2
import numpy as np
import pandas as pd
import argparse
import os


def get_flow_vectors(
    video_path, target_frame_numbers, step=20, magnitude_thresh=1.0, length_factor=10
):
    cap = cv2.VideoCapture(video_path)
    ret, prev_color = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Failed to read the first frame from the video.")
    prev_gray = cv2.cvtColor(prev_color, cv2.COLOR_BGR2GRAY)

    target_frame_numbers_set = set(target_frame_numbers)
    results = {}
    frame_number = 1
    while True:
        ret, curr_color = cap.read()
        if not ret:
            print("End of video reached.")
            break
        curr_gray = cv2.cvtColor(curr_color, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        if frame_number in target_frame_numbers_set:
            flow_trails_image = curr_color.copy()
            h, w = curr_color.shape[:2]
            flow_vectors = []

            for y in range(0, h, step):
                for x in range(0, w, step):
                    fxy = flow[y, x]
                    mag = np.sqrt(fxy[0] ** 2 + fxy[1] ** 2)
                    if mag < magnitude_thresh:
                        continue

                    angle = np.arctan2(fxy[1], fxy[0]) * 180.0 / np.pi
                    if angle < 0:
                        angle += 360

                    hsv = np.uint8([[[angle / 2, 255, 255]]])
                    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                    color = tuple(int(c) for c in bgr[0, 0])

                    start = (x, y)
                    end = (
                        int(x + fxy[0] * length_factor),
                        int(y + fxy[1] * length_factor),
                    )
                    cv2.line(flow_trails_image, start, end, color, 1, cv2.LINE_AA)
                    cv2.circle(flow_trails_image, start, 1, color, -1)

                    flow_vectors.append([start, end])

            results[frame_number] = (flow_trails_image, flow_vectors)
            print(
                f"Extracted {len(flow_vectors)} flow vectors for frame {frame_number}."
            )

            target_frame_numbers_set.remove(frame_number)

            if not target_frame_numbers_set:
                print("All target frames processed.")
                break

        prev_gray = curr_gray
        frame_number += 1

    cap.release()
    return results


def main():
    parser = argparse.ArgumentParser(description="Compute optical flow vectors.")
    parser.add_argument("video_path", help="Path to the video file.")
    parser.add_argument("start_frame", type=int, help="Start frame number.")
    parser.add_argument("end_frame", type=int, help="End frame number.")
    parser.add_argument("output_dir", help="Base directory to save outputs.")

    args = parser.parse_args()

    target_frame_numbers = np.arange(args.start_frame, args.end_frame + 1)

    results = get_flow_vectors(
        args.video_path,
        target_frame_numbers,
        step=15,
        magnitude_thresh=0.1,
        length_factor=20,
    )

    trails_dir = os.path.join(args.output_dir, "trails")
    vectors_dir = os.path.join(args.output_dir, "vectors")
    os.makedirs(trails_dir, exist_ok=True)
    os.makedirs(vectors_dir, exist_ok=True)
    for frame_number, (flow_trails_image, vectors) in results.items():
        image_path = os.path.join(
            trails_dir, f"flow_visualization_frame_{frame_number}.png"
        )
        cv2.imwrite(image_path, flow_trails_image)
        print(f"Saved image for frame {frame_number}.")

        start_points = [v[0] for v in vectors]
        end_points = [v[1] for v in vectors]
        frame_df = pd.DataFrame(
            {
                "start_x": [x for x, y in start_points],
                "start_y": [y for x, y in start_points],
                "end_x": [x for x, y in end_points],
                "end_y": [y for x, y in end_points],
            }
        )
        csv_path = os.path.join(vectors_dir, f"flow_vectors_frame_{frame_number}.csv")
        frame_df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
