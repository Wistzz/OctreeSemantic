import numpy as np
from plyfile import PlyData
import open3d as o3d


def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))


def visualize_ply(ply_path):
    # Load the PLY file
    ply_data = PlyData.read(ply_path)
    vertex_data = ply_data['vertex'].data

    # Extract the point cloud attributes
    points = np.array([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
    colors = np.array([vertex_data['red'], vertex_data['green'], vertex_data['blue']]).T / 255.0
    opacity = vertex_data['opacity']

    # Apply the opacity filter
    sigmoid_opacity = sigmoid(opacity)
    filtered_indices = sigmoid_opacity >= 0.1
    filtered_points = points[filtered_indices]
    filtered_colors = colors[filtered_indices]

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    # Create a visualizer in headless mode
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)

    # Set up rendering options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.0  # You can adjust the point size

    # Update the geometry and render
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    # Save the rendered image
    output_image_path = "point_cloud_render.png"
    vis.capture_screen_image(output_image_path)
    vis.destroy_window()


if __name__ == "__main__":
    # Replace with the path to your PLY file
    ply_path = "/data/sunwei/OctreeSemantic/output/0.02_figurines/point_cloud/iteration_40000/point_cloud.ply"  # Replace with the actual path to your PLY file
    visualize_ply(ply_path)

# if __name__ == "__main__":
#     ply_path = "/data/sunwei/OctreeSemantic/output/0.02_figurines/point_cloud/iteration_40000/point_cloud.ply"  # Replace with the actual path to your PLY file
#     output_pcd_path = "output_point_cloud.pcd"  # Replace with the desired output path
#     save_pcd(ply_path, output_pcd_path)

