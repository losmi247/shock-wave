import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_angle(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    return np.degrees(angle_rad)


# Calculate pairwise angles in each frame
angles = []
for frame_num in range(448, 449, 1):
    df = pd.read_csv(
        f"./results/transformed_vectors/transformed_flow_vectors_frame_{frame_num}.csv"
    )
    vectors = df[["end_x", "end_y"]].values - df[["start_x", "start_y"]].values

    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            angles.append(get_angle(vectors[i], vectors[j]))

    print(f"Finished frame {frame_num}")

# Plot histogram of angles
plt.figure(figsize=(8, 6))
plt.hist(angles, bins=50, edgecolor="black")
plt.title("Histogram of angles between BEV flow vectors")
plt.xlabel("Angle (degrees)")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig(
    "./results/angle_plots/bev_flow_vectors_angles", dpi=300, bbox_inches="tight"
)
