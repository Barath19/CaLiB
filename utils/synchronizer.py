#!/usr/bin/env python

#####################################################
# This code is part of my RnD as part of my course
# Contact: barath19 (github)
#####################################################
import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
import cv2
import os
import time
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
import numpy as np

# ROS node initialization
rospy.init_node('lidar_and_cam_synchronizer')
nh = rospy.get_namespace()

# Get parameters or use default values
root_output_dir = rospy.get_param("~root_output_dir", "/home/bk/rnd/HBRS-dataset/outputs")
debug = rospy.get_param("~debug", False)

# Create subscribers
image_sub = Subscriber('/cam1/color/image_raw', Image)
camera_info_sub = Subscriber('/cam1/color/camera_info', CameraInfo)
point_cloud_sub = Subscriber('/cam2/depth/color/points', PointCloud2)

# Create synchronizer
sync = ApproximateTimeSynchronizer([image_sub, camera_info_sub, point_cloud_sub], queue_size=10, slop=0.1)

def callback(image, camera_info, point_cloud):
    print("All data in sync!")

    # Create a date string from the point cloud's timestamp for the file name
    timestamp = point_cloud.header.stamp
    timestamp_str = timestamp.to_nsec()
    fractional_second_digits = 4

    # Convert the ROS image to an OpenCV image
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")

    # Update GUI Window
    if debug:
        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)

    # Save the image
    output_file = os.path.join(camera_output_dir, f"camera_{timestamp_str:.0f}_{timestamp.nsecs:0{fractional_second_digits}d}.jpg")
    cv2.imwrite(output_file, cv_image)

    # Convert the point data to a numpy array
    points = np.array(list(pc2.read_points(point_cloud, field_names=("x", "y", "z", "intensity"), skip_nans=True)))

    # Save the point cloud as a .pcd file
    output_file = os.path.join(lidar_output_dir, f"lidar_{timestamp_str:.0f}_{timestamp.nsecs:0{fractional_second_digits}d}.pcd")
    with open(output_file, "w") as f:
        f.write("# .PCD v0.7 - Point Cloud Data\n")
        f.write(f"VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH {points.shape[0]}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {points.shape[0]}\nDATA ascii\n")
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]} {point[3]}\n")

    print(output_file)

sync.registerCallback(callback)

# Create the folder for the current run
output_dir = os.path.join(root_output_dir, time.strftime("run_%Y_%m_%d_%H_%M_%S", time.localtime()))
lidar_output_dir = os.path.join(output_dir, "pcd")
camera_output_dir = os.path.join(output_dir, "jpg")

os.makedirs(lidar_output_dir, exist_ok=True)
os.makedirs(camera_output_dir, exist_ok=True)

print(lidar_output_dir)

# Start ROS spin
rospy.spin()
