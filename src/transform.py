import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os


def select_roi_points(image, num_points=4):
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < num_points:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select ROI Points", image)

    print(f"Click {num_points} points in clockwise order.")
    cv2.imshow("Select ROI Points", image)
    cv2.setMouseCallback("Select ROI Points", click_event)

    while len(points) < num_points:
        cv2.waitKey(1)

    cv2.destroyWindow("Select ROI Points")
    return np.array(points, dtype=np.float32)


def transform_point_to_birds_eye_view(homography, point):
    pt_h = np.array([point[0], point[1], 1.0])
    pt_transformed = homography @ pt_h
    pt_transformed /= pt_transformed[2]
    return pt_transformed[:2]


def map_vectors_to_birds_eye_view(homography, vectors):
    transformed_vectors = []
    for start, end in vectors:
        start_bev = transform_point_to_birds_eye_view(homography, start)
        end_bev = transform_point_to_birds_eye_view(homography, end)
        transformed_vectors.append((start_bev, end_bev))

    return transformed_vectors


def get_transformed_vectors(
    vectors_dir, frame_number, homography, bev_width, bev_height
):
    csv_path = os.path.join(vectors_dir, f"flow_vectors_frame_{frame_number}.csv")
    df = pd.read_csv(csv_path)
    vectors = [((x1, y1), (x2, y2)) for x1, y1, x2, y2 in df.values]

    bev_vectors = map_vectors_to_birds_eye_view(homography, vectors)

    filtered_vectors = []
    for start, end in bev_vectors:
        if (
            0 <= start[0] < bev_width
            and 0 <= start[1] < bev_height
            and 0 <= end[0] < bev_width
            and 0 <= end[1] < bev_height
        ):
            filtered_vectors.append((start, end))

    return filtered_vectors


def save_transformed_trails_image(
    trails_dir, transformed_trails_dir, frame_number, homography, bev_width, bev_height
):
    trails_image_path = os.path.join(
        trails_dir, f"flow_visualization_frame_{frame_number}.png"
    )
    trails_image = cv2.imread(trails_image_path)
    if trails_image is not None:
        warped_image = cv2.warpPerspective(
            trails_image, homography, dsize=(bev_width, bev_height)
        )

        transformed_trails_image_path = os.path.join(
            transformed_trails_dir, f"warped_frame_{frame_number}.png"
        )
        cv2.imwrite(transformed_trails_image_path, warped_image)
    else:
        print(f"Warning: Could not load trails image for frame {frame_number}")


def save_vectors_plot(transformed_vectors_plots_dir, frame_number, vectors):
    plt.figure(figsize=(8, 8))
    for start, end in vectors:
        vector = [end[0] - start[0], end[1] - start[1]]
        norm = np.linalg.norm(vector)
        if norm > 1e-8:
            vector = vector / norm

        if abs(start[0]) > 1000 or abs(start[1]) > 1000:
            continue

        plt.arrow(
            start[0],
            start[1],
            vector[0] * 20,
            vector[1] * 20,
            head_width=2,
            length_includes_head=True,
        )

    plt.title(f"Bird's eye view of flow vectors - frame {frame_number}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.axis("equal")

    plot_path = os.path.join(transformed_vectors_plots_dir, f"frame_{frame_number}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Transform flow vectors to bird's eye view."
    )
    parser.add_argument(
        "output_dir", help="Path to the corresponding video file's output directory."
    )
    parser.add_argument("start_frame", type=int, help="Start frame number.")
    parser.add_argument("end_frame", type=int, help="End frame number.")
    parser.add_argument(
        "--bev_width", type=int, default=300, help="Width of the bird's eye view image."
    )
    parser.add_argument(
        "--bev_height",
        type=int,
        default=400,
        help="Height of the bird's eye view image.",
    )

    args = parser.parse_args()

    target_frame_numbers = np.arange(args.start_frame, args.end_frame + 1)

    first_trails_frame = os.path.join(
        args.output_dir, "trails", f"flow_visualization_frame_{args.start_frame}.png"
    )
    image_for_roi = cv2.imread(first_trails_frame)
    if image_for_roi is None:
        raise FileNotFoundError(
            f"Could not load {first_trails_frame} for ROI selection."
        )

    perspective_points = select_roi_points(image_for_roi)
    # perspective_points = np.array(
    #    [[142, 118], [225, 98], [467, 134], [277, 222]], dtype=np.float32
    # )
    dest_points = np.array(
        [
            [0, 0],
            [args.bev_width, 0],
            [args.bev_width, args.bev_height],
            [0, args.bev_height],
        ],
        dtype=np.float32,
    )
    homography_matrix = cv2.getPerspectiveTransform(perspective_points, dest_points)

    # Transform each frame's flow vectors to bird's eye view
    vectors_dir = os.path.join(args.output_dir, "vectors")
    trails_dir = os.path.join(args.output_dir, "trails")
    transformed_vectors_dir = os.path.join(args.output_dir, "transformed_vectors")
    transformed_trails_dir = os.path.join(args.output_dir, "transformed_trails")
    transformed_vectors_plots_dir = os.path.join(
        args.output_dir, "transformed_vectors_plots"
    )
    os.makedirs(transformed_vectors_dir, exist_ok=True)
    os.makedirs(transformed_trails_dir, exist_ok=True)
    os.makedirs(transformed_vectors_plots_dir, exist_ok=True)

    for frame_number in target_frame_numbers:
        transformed_vectors = get_transformed_vectors(
            vectors_dir,
            frame_number,
            homography_matrix,
            args.bev_width,
            args.bev_height,
        )

        transformed_data = []
        for start_w, end_w in transformed_vectors:
            transformed_data.append(
                {
                    "start_x": start_w[0],
                    "start_y": start_w[1],
                    "end_x": end_w[0],
                    "end_y": end_w[1],
                }
            )
        transformed_df = pd.DataFrame(
            transformed_data, columns=["start_x", "start_y", "end_x", "end_y"]
        )
        df_path = os.path.join(
            transformed_vectors_dir,
            f"transformed_flow_vectors_frame_{frame_number}.csv",
        )
        transformed_df.to_csv(df_path, index=False)
        print(f"Saved transformed vectors for frame {frame_number}")

        save_transformed_trails_image(
            trails_dir,
            transformed_trails_dir,
            frame_number,
            homography_matrix,
            args.bev_width,
            args.bev_height,
        )

        save_vectors_plot(
            transformed_vectors_plots_dir, frame_number, transformed_vectors
        )


if __name__ == "__main__":
    main()
