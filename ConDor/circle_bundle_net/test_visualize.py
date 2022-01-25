import numpy as np
import tensorflow as tf
import open3d as o3d
import h5py
from circle_bundle_net.local_directions import local_pca, local_direction, center_orientation
from group_points import GroupPoints
from pooling import kdtree_indexing, aligned_kdtree_indexing, kdtree_indexing_, aligned_kdtree_indexing_, kd_pooling_1d

"""
print("Load a ply point cloud, print it, and render it")
# pcd = o3d.io.read_point_cloud("E:/Users/Adrien/Documents/Datasets/ModelNet40_hdf5/modelnet40_hdf5_1024_classes/tests/test_toilet_chair_0.obj")
xyz = np.random.rand(1024, 3)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
print(pcd)
# print(np.asarray(pcd.points))
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
"""
"""
if __name__ == "__main__":

    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud("../../TestData/fragment.ply")
    o3d.visualization.draw_geometries([pcd])

    print("Let's draw some primitives")
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0,
                                                    height=1.0,
                                                    depth=1.0)
    mesh_box.compute_vertex_normals()
    mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
    mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.3,
                                                              height=4.0)
    mesh_cylinder.compute_vertex_normals()
    mesh_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[-2, -2, -2])

    print("We draw a few primitives using collection.")
    o3d.visualization.draw_geometries(
        [mesh_box, mesh_sphere, mesh_cylinder, mesh_frame])

    print("We draw a few primitives using + operator of mesh.")
    o3d.visualization.draw_geometries(
        [mesh_box + mesh_sphere + mesh_cylinder + mesh_frame])

    print("Let's draw a cubic using o3d.geometry.LineSet.")
    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([line_set])


    print("Let's draw a textured triangle mesh from obj file.")
    textured_mesh = o3d.io.read_triangle_mesh("../../TestData/crate/crate.obj")
    textured_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([textured_mesh]
"""

def draw_oriented_pointcloud(x, n, t=1.0):
    a = x
    b = x + t*n
    points = []
    lines = []
    for i in range(a.shape[0]):
        ai = [a[i, 0], a[i, 1], a[i, 2]]
        bi = [b[i, 0], b[i, 1], b[i, 2]]
        points.append(ai)
        points.append(bi)
        lines.append([2*i, 2*i+1])
    colors = [[1, 0, 0] for i in range(len(lines))]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(a)

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([line_set, pcd])
    # o3d.visualization.draw_geometries([pcd])
    pass


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    print(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    normals = f['normal'][:]
    return data, label, normals


X, L, N = load_h5("E:/Users/Adrien/Documents/Datasets/ModelNet40_hdf5/modelnet40_hdf5_1024_normals/data_hdf5/train_data_1.h5")
x = tf.expand_dims(X[58, ...], axis=0)
n = tf.expand_dims(N[58, ...], axis=0)


x, idx, _ = kdtree_indexing_(x)
n = tf.gather_nd(x, idx)
num_points = [1024, 256, 64]



# draw_oriented_pointcloud(x, n, t=0.1)
points = [x]
normals = [n]
for i in range(len(num_points)-1):
    points.append(kd_pooling_1d(points[-1], int(num_points[i]/num_points[i+1])))
    ni = kd_pooling_1d(normals[-1], int(num_points[i]/num_points[i+1]))
    ni = tf.linalg.l2_normalize(ni, axis=-1)
    normals.append(ni)
patches = []
for i in range(len(num_points)-1):
    gi = GroupPoints(radius=1., patch_size_source=32)({"source points": points[i], "target points": points[i+1]})
    patches.append(gi["patches source"])

i = 0
x = points[i+1]
n, line, plate, blob_mask = local_direction(patches[i], x, weights=None, center=True)
# e, v = local_pca(patches[i])
# n = v[..., 2]

# n = normals[i+1]
print(n.shape)
n = np.array(n[0, ...])
x = np.array(x[0, ...])


draw_oriented_pointcloud(x, n, t=0.1)


