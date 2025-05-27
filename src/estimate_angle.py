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
for frame_num in range(420, 500, 1):
    df = pd.read_csv(f"./results/vectors/flow_vectors_frame_{frame_num}.csv")
    vectors = (
        df[["x_world_end", "z_world_end"]].values
        - df[["x_world_start", "z_world_start"]].values
    )

    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            angles.append(get_angle(vectors[i], vectors[j]))

# Plot histogram of angles
plt.figure(figsize=(8, 6))
plt.hist(angles, bins=50, edgecolor="black")
plt.title("Histogram of angles between motion vectors")
plt.xlabel("Angle (degrees)")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("./results/angle_plots/angles1", dpi=300, bbox_inches="tight")
