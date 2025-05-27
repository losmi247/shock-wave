import numpy as np
import math
import scipy
import pandas as pd
import scipy.spatial
import matplotlib.pyplot as plt


#
# Given the 3x3 camera intrinsics matrix K, 3x3 rotation R,
# and the 3x1 camera location C in world, returns the
# homography from world coordinate system to image coordinate
# system for the plane y_world = 0.
#
def compute_homography(K, R, C):
    Rt = np.column_stack((R[:, 0], R[:, 2], -R @ C))
    return K @ Rt


#
# Applies the given inverse homography to map the image plane
# point to the corresponding world plane point.
#
def transform_image_to_world(homography_inv, point_image):
    point_img_h = np.array([point_image[0], point_image[1], 1.0])
    point_world_h = homography_inv @ point_img_h
    point_world_h /= point_world_h[2]
    return point_world_h[:2]


#
# Maps the given collection of image plane vectors to
# corresponding world plane vectors in the y=0 plane.
#
def map_vectors_to_world(vectors, K, R, C):
    H = compute_homography(K, R, C)
    H_inv = np.linalg.inv(H)

    world_vectors = []
    for start, end in vectors:
        start_world = transform_image_to_world(H_inv, start)
        end_world = transform_image_to_world(H_inv, end)
        world_vectors.append([start_world, end_world])

    return world_vectors


# Camera and rotation setup
camera_in_world = [0, 5, 0]

image_width = 480
image_height = 224
fov_h = math.pi / 3
f = image_width / (2 * math.tan(fov_h / 2))
camera_intrinsics = np.array(
    [[f, 0, image_width / 2], [0, f, image_height / 2], [0, 0, 1]]
)

theta_x = 10
rot_x = scipy.spatial.transform.Rotation.from_euler(
    "x", -theta_x, True
).as_matrix()  # negative theta_x as it's a left-hand coordinate system
theta_y = 45
rot_y = scipy.spatial.transform.Rotation.from_euler("y", theta_y, True).as_matrix()
rotations = rot_y @ rot_x

print(
    f"Homography: {compute_homography(camera_intrinsics, rotations, camera_in_world)}"
)


# Process each frame
frames = np.arange(420, 500, 1)

for frame_id in frames:
    df = pd.read_csv(f"./results/vectors/flow_vectors_frame_{frame_id}.csv")

    frame_vectors = df.values
    vectors = [((x1, y1), (x2, y2)) for x1, y1, x2, y2 in frame_vectors]

    world_vectors = map_vectors_to_world(
        vectors, camera_intrinsics, rotations, camera_in_world
    )

    transformed_data = []
    for start_w, end_w in world_vectors:
        transformed_data.append(
            {
                "x_world_start": start_w[0],
                "z_world_start": start_w[1],
                "x_world_end": end_w[0],
                "z_world_end": end_w[1],
            }
        )

    df_world = pd.DataFrame(transformed_data)
    df_world.to_csv(
        f"./results/world_vectors/world_flow_vectors_frame_{frame_id}.csv", index=False
    )

    plt.figure(figsize=(8, 8))
    for _, row in df_world.iterrows():
        plt.arrow(
            row["x_world_start"],
            row["z_world_start"],
            row["x_world_end"] - row["x_world_start"],
            row["z_world_end"] - row["z_world_start"],
            head_width=0.2,
            length_includes_head=True,
        )
    plt.title(f"World Vectors - Frame {frame_id}")
    plt.xlabel("X (world)")
    plt.ylabel("Z (world)")
    plt.grid(True)
    plt.axis("equal")
    plt.savefig(
        f"./results/world_vectors_figures/frame_{frame_id}.png",
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Finished frame {frame_id}")
