import open3d as o3d

pcd = o3d.io.read_point_cloud("/home/bk/Study/RnD/CaLiB/Fixed_file.ply")
o3d.io.write_point_cloud("/home/bk/Study/RnD/CaLiB/lidar1.pcd", pcd)

'''
from plyfile import PlyData, PlyProperty, PlyElement
from pathlib import Path

# read in file
data = PlyData.read("/home/bk/Study/RnD/CaLiB/data/lidar.ply")
# make a new PlyElement with type "vertex" with our existing data
renamed_element = PlyElement.describe(data.elements[0].data, 'vertex',
                        comments=[f'Renamed from: {data.elements[0].name}'])
# Make a new PlyData object of binary format 
fixed_data = PlyData([renamed_element], text=True)
# Write it out
fixed_data.write("Fixed_file.ply")

'''