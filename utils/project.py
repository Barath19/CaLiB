import cv2
import pcl
import numpy as np
import matplotlib.pyplot as plt

# Read PCD file
cloud = pcl.load_XYZI("data/lidar_lab.pcd")

image = cv2.imread("data/camera_lab.jpg")

# Camera calibration matrices
K = np.array([[613.2254638671875, 0, 338.2792663574219],
              [0, 613.3134155273438, 250.49005126953125],
              [0, 0, 1]])

dist_coeff = np.array([0, 0, 0, 0, 0], dtype=float)  # Distortion coefficients

#extrinsic_matrix = np.float64([[0.038128, -0.999234, 0.00881841, 0.190658], 
#                               [-0.0128125, -0.00931295, -0.999875, -0.546559], 
#                               [0.999191, 0.0380102, -0.131577, -0.171036], 
#                               [0.00000, 0.00000, 0.00000, 1.00000]])
extrinsic_matrix = np.float64([[0.123581, -0.991935, 0.0281621, 0.126832], 
                               [0.00353999, -0.0279388, -0.999603, -0.516982], 
                               [0.992328, 0.123632, 0.0000, -0.00730229], 
                               [0.00000, 0.00000, 0.00000, 1.00000]])

color_cloud = np.array(list(cloud))# Read image
intensity = color_cloud[:, 3]
cmap = plt.cm.get_cmap('hsv')
colors = cmap(intensity / np.max(intensity))
rgb = tuple(map(tuple, np.array(colors[:, :3] * 255).astype(int)))

# Project 3D points onto 2D image
for i, point in enumerate(cloud):
    # Extract X, Y, Z, and Intensity coordinates from the point cloud
    x, y, z, intensity = point[0], point[1], point[2], point[3]

    # Homogeneous 3D coordinates
    point_3d = np.array([[x, y, z, 1]])

    # Transform 3D point to world coordinates using extrinsic matrix
    world_coordinates = np.dot(extrinsic_matrix, point_3d.T).T

    # Project world coordinates to 2D image coordinates using intrinsic matrix
    point_2d = cv2.projectPoints(world_coordinates[:, :3], (0, 0, 0), (0, 0, 0), K, dist_coeff)[0][0][0]


    # Draw the point on the image with color
    cv2.circle(image, (int(point_2d[0]), int(point_2d[1])), radius=2, color=(int(rgb[i][0]),int(rgb[i][1]),int(rgb[i][2])), thickness=-1)
cv2.imshow("Projected Point Cloud", image)
cv2.imwrite("data/point_cloud.png", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

