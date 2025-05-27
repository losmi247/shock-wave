import cv2
import numpy as np
import pandas as pd


def compute_flow_vectors(
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
            vis = curr_color.copy()
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
                    cv2.line(vis, start, end, color, 1, cv2.LINE_AA)
                    cv2.circle(vis, start, 1, color, -1)

                    flow_vectors.append([start, end])

            results[frame_number] = (vis, flow_vectors)
            print(
                f"Extracted vectors for frame {frame_number} ({len(flow_vectors)} vectors)"
            )

            target_frame_numbers_set.remove(frame_number)

            if not target_frame_numbers_set:
                print("All target frames processed.")
                break

        prev_gray = curr_gray
        frame_number += 1

    cap.release()
    return results


video_path = "./videos/4.mp4"
target_frame_numbers = np.arange(420, 500, 1)

results = compute_flow_vectors(
    video_path, target_frame_numbers, step=15, magnitude_thresh=0.1, length_factor=20
)

for frame_number, (vis_image, vectors) in results.items():
    cv2.imwrite(
        "./results/trails/flow_visualization_frame_{}.png".format(frame_number),
        vis_image,
    )
    print(f"Saved image for frame {frame_number}. First 5 vectors: {vectors[:5]}")

    frame_df = pd.DataFrame(vectors, columns=["start", "end"])
    frame_df[["start_x", "start_y"]] = pd.DataFrame(
        frame_df["start"].tolist(), index=frame_df.index
    )
    frame_df[["end_x", "end_y"]] = pd.DataFrame(
        frame_df["end"].tolist(), index=frame_df.index
    )
    frame_df = frame_df.drop(columns=["start", "end"])
    frame_df.to_csv(
        "./results/vectors/flow_vectors_frame_{}.csv".format(frame_number), index=False
    )
