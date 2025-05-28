import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import argparse
import os


def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-8 else v


def get_angle(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    return np.degrees(angle_rad)


def get_dominant_directions(clustering_dir, frame_number, vectors_df):
    directions = []
    starts = []

    for _, row in vectors_df.iterrows():
        direction = np.array(
            [row["end_x"] - row["start_x"], row["end_y"] - row["start_y"]]
        )
        if np.linalg.norm(direction) > 1e-3:
            directions.append(normalize(direction))
            starts.append([row["start_x"], row["start_y"]])

    if len(directions) < 10:
        return None
    directions = np.array(directions)
    starts = np.array(starts)

    kmeans = KMeans(n_clusters=2, n_init="auto").fit(directions)
    vector_labels = kmeans.labels_

    dominant_cluster_directions = {}
    for i in range(2):
        cluster_directions = directions[vector_labels == i]
        dominant_cluster_directions[i] = normalize(np.mean(cluster_directions, axis=0))

    # Plot this frame's dominant directions on top of the vectors
    colors = ["r", "b"]
    plt.figure(figsize=(8, 8))

    for i, (start, direction, label) in enumerate(
        zip(starts, directions, vector_labels)
    ):
        plt.arrow(
            start[0],
            start[1],
            direction[0] * 20,
            direction[1] * 20,
            head_width=2,
            color=colors[label],
            length_includes_head=True,
        )

    for i in range(2):
        plt.arrow(
            0,
            0,
            dominant_cluster_directions[i][0] * 100,
            dominant_cluster_directions[i][1] * 100,
            color=colors[i],
            width=5,
            head_width=10,
            label=f"Mean Dir {i+1}",
            length_includes_head=True,
        )

    plt.title(f"Flow vectors and dominant directions - frame {frame_number}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.savefig(
        os.path.join(clustering_dir, f"dominant_directions_frame_{frame_number}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return dominant_cluster_directions


def main():
    parser = argparse.ArgumentParser(
        description="Cluster vectors in each frame to get dominant directions."
    )
    parser.add_argument(
        "output_dir", help="Path to the corresponding video file's output directory."
    )
    parser.add_argument("start_frame", type=int, help="Start frame number.")
    parser.add_argument("end_frame", type=int, help="End frame number.")

    args = parser.parse_args()

    target_frame_numbers = np.arange(args.start_frame, args.end_frame + 1)

    transformed_vectors_dir = os.path.join(args.output_dir, "transformed_vectors")
    clustering_dir = os.path.join(args.output_dir, "clustering")
    os.makedirs(clustering_dir, exist_ok=True)

    cone_tip_angles = []
    for frame_number in target_frame_numbers:
        csv_path = os.path.join(
            transformed_vectors_dir,
            f"transformed_flow_vectors_frame_{frame_number}.csv",
        )
        df = pd.read_csv(csv_path)

        dominant_directions = get_dominant_directions(clustering_dir, frame_number, df)
        if dominant_directions is None:
            continue

        angle = get_angle(dominant_directions[0], dominant_directions[1])
        cone_tip_angle = 180 - angle
        print(f"(frame {frame_number}): cone tip angle {cone_tip_angle} deg")
        cone_tip_angles.append(cone_tip_angle)

    # Plot the sampled cone tip angles
    mean_cone_tip_angle = np.mean(cone_tip_angles)
    std_cone_tip_angle = np.std(cone_tip_angles)

    plt.figure(figsize=(10, 6))
    plt.hist(cone_tip_angles, bins=10, color="gray", edgecolor="black", alpha=0.7)

    plt.axvline(
        mean_cone_tip_angle,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_cone_tip_angle:.2f}°",
    )
    plt.axvline(
        mean_cone_tip_angle - std_cone_tip_angle,
        color="blue",
        linestyle="--",
        linewidth=1.5,
        label=f"±1 Std Dev = {std_cone_tip_angle:.2f}°",
    )
    plt.axvline(
        mean_cone_tip_angle + std_cone_tip_angle,
        color="blue",
        linestyle="--",
        linewidth=1.5,
    )

    plt.title("Histogram of estimated cone tip angles")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(args.output_dir, "cone_tip_angle_histogram.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    main()
