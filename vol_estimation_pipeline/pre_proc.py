
import numpy as np
import open3d as o3d
import math
num_points=2048
def process(area, data):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    R = np.identity(3)
    extent = np.ones(3) / 1.5
    center = np.mean(data, axis=0)
    obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
    pcd = pcd.crop(obb)
    para = 0.017
    if area < 100:
        para = 0.013
    elif area < 200:
        para = 0.015
    else:
        para = 0.017
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
    plane_model, inliers = pcd.segment_plane(distance_threshold=para,
                                             ransac_n=3,
                                             num_iterations=10000)

    [a, b, c, d] = plane_model
    plane_pcd = pcd.select_by_index(inliers)
    plane_pcd.paint_uniform_color([1.0, 0, 0])
    object_pcd = pcd.select_by_index(inliers, invert=True)
    object_pcd.paint_uniform_color([0, 0, 1.0])

    #o3d.visualization.draw_geometries([object_pcd])
    object_pcd = object_pcd.translate((0, 0, d / c))

    cos_theta = c / math.sqrt(a ** 2 + b ** 2 + c ** 2)
    sin_theta = math.sqrt((a ** 2 + b ** 2) / (a ** 2 + b ** 2 + c ** 2))
    u_1 = b / math.sqrt(a ** 2 + b ** 2)
    u_2 = -a / math.sqrt(a ** 2 + b ** 2)

    rotation_matrix = np.array(
        [[cos_theta + u_1 ** 2 * (1 - cos_theta), u_1 * u_2 * (1 - cos_theta), u_2 * sin_theta],
         [u_1 * u_2 * (1 - cos_theta), cos_theta + u_2 ** 2 * (1 - cos_theta), -u_1 * sin_theta],
         [-u_2 * sin_theta, u_1 * sin_theta, cos_theta]])

    object_pcd.rotate(rotation_matrix)
    object_pcd = object_pcd.voxel_down_sample(voxel_size=0.001)
    labels = np.array(object_pcd.cluster_dbscan(eps=0.01, min_points=10, print_progress=True))

    candidates = [len(np.where(labels == j)[0]) for j in np.unique(labels)]
    best_candidate = int(np.unique(labels)[np.where(candidates == np.max(candidates))[0]])
    max_label = labels.max()
    object_pcd = object_pcd.select_by_index(list(np.where(labels == best_candidate)[0]))
    if np.asarray(object_pcd.points).shape[0] > 2048:
        object_pcd = object_pcd.farthest_point_down_sample(2048)

    hull, _ = object_pcd.compute_convex_hull()
    hull.compute_vertex_normals()
    pcd = hull.sample_points_uniformly(num_points * 3)
    pcd = hull.sample_points_poisson_disk(number_of_points=num_points, pcl=pcd)

    cur_points = np.asarray(pcd.points)
    points = cur_points
    centroid = np.mean(points, axis=0)
    points -= centroid
    furthest_distance = np.max(np.linalg.norm(points))
    points /= furthest_distance
    points=np.asarray(points)
    return points