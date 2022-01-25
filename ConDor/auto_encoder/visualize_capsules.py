import matplotlib.pyplot as plt
# import torch
import tensorflow as tf
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import open3d as o3d
import colorsys
import seaborn as sns

from utils.data_prep_utils import load_h5_data
from pooling import GridBatchSampler
from utils.pointclouds_utils import setup_pcl_viewer
import vispy

from spherical_harmonics.kernels import tf_eval_monom_basis, tf_monom_basis_offset, tf_monomial_basis_3D_idx, compute_monomial_basis_offset
from spherical_harmonics.kernels import real_spherical_harmonic, zernike_kernel_3D, tf_zernike_kernel_basis
from spherical_harmonics.kernels import A, B, associated_legendre_polynomial
from utils.pointclouds_utils import generate_3d
from group_points import GroupPoints
from sparse_grid_sampling_eager import GridSampler, GridPooling, extract_batch_idx
from pooling import kd_pooling_1d, kd_median_sampling, kdtree_indexing, aligned_kdtree_indexing
from utils.pointclouds_utils import np_kd_tree_idx, pc_batch_preprocess
from data_providers.provider import Provider
from data_providers.classification_datasets import datsets_list
from spherical_harmonics.clebsch_gordan_decomposition import tf_clebsch_gordan_decomposition
from pooling import extract_samples_slices
from time import time
from sklearn.neighbors import NearestNeighbors
# from unocs.train import lexicographic_ordering

from auto_encoder.tfn_capsules import TFN
from pooling import kdtree_indexing, aligned_kdtree_indexing, kdtree_indexing_, aligned_kdtree_indexing_, kd_pooling_1d
from plyfile import PlyData, PlyElement

import h5py

def create_color_samples(N):
    '''
    Creates N distinct colors
    N x 3 output
    '''

    palette = sns.color_palette(None, N)
    palette = np.array(palette)

    return palette

"""
def convert_tensor_2_numpy(x):
    '''
    Convert pytorch tensor to numpy
    '''

    out = x.detach().cpu().squeeze(0).numpy()

    return out
"""

def save_pointcloud(x, filename="./pointcloud.ply"):
    '''
    Save point cloud to the destination given in the filename
    x can be list of inputs or numpy array of N x 3
    '''

    label_map = []
    if isinstance(x, list):

        pointcloud = []
        labels = create_color_samples(len(x))
        for i in range(len(x)):
            pts = np.array(x[i])
            # print(pts.shape, "vis")
            # pts = convert_tensor_2_numpy(pts).transpose((1, 0))



            pointcloud.append(pts)
            label_map.append(np.repeat(labels[i:(i + 1)], pts.shape[0], axis=0))

        # x = np.concatenate(x, axis = 0)
        pointcloud = np.concatenate(pointcloud, axis=0)
        x = pointcloud.copy()
        label_map = np.concatenate(label_map, axis=0)
    else:
        x = np.array(x)
        # print(x.shape)
        label_map = np.ones((len(x), 3)) * 0.5

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x)
    pcd.colors = o3d.utility.Vector3dVector(label_map)

    o3d.io.write_point_cloud(filename, pcd)


"""
def vis_pts(x, att=None, vis_fn="temp.png"):
    idx = 0
    pts = x[idx].transpose(1, 0).cpu().numpy()
    label_map = None
    if att is not None:
        label_map = torch.argmax(att, dim=2)[idx].squeeze().cpu().numpy()
        label_map 
    vis_pts_att(pts, label_map, fn=vis_fn)
"""

def vis_recon(pc_recon, vis_fn="temp.png", idx=0):
    # pc_recon: a list of sets of points
    # idx = 0
    label_map = []
    pts = []
    for i, patch in enumerate(pc_recon):
        pts_cur = patch[idx].transpose(1, 0).cpu().numpy()
        label_map += [np.ones(len(pts_cur)) * i]
        pts += [pts_cur]

    pts = np.concatenate(pts, axis=0)
    label_map = np.concatenate(label_map, axis=0)
    vis_pts_att(pts, label_map, fn=vis_fn)

def onehot_to_int(y):
    y = tf.argmax(y, axis=-1)
    y = np.array(y, dtype=np.int32)
    return y


def vis_pts_att(pts, label_map, fn="temp.png", marker=".", alpha=0.9):
    # pts (n, d): numpy, d-dim point cloud
    # label_map (n, ): numpy or None
    # fn: filename of visualization
    assert pts.shape[1] in [2, 3]
    if pts.shape[1] == 2:
        xs = pts[:, 0]
        ys = pts[:, 1]
        if label_map is not None:
            plt.scatter(xs, ys, c=label_map, cmap="jet", marker=".", alpha=0.9, edgecolor="none")
        else:
            plt.scatter(xs, ys, c="grey", alpha=0.8, edgecolor="none")
        # save
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.axis("off")
    elif pts.shape[1] == 3:
        TH = 0.7
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_zlim(-TH, TH)
        ax.set_xlim(-TH, TH)
        ax.set_ylim(-TH, TH)
        xs = pts[:, 0]
        ys = pts[:, 1]
        zs = pts[:, 2]
        if label_map is not None:
            ax.scatter(xs, ys, zs, c=label_map, cmap="jet", marker=marker, alpha=alpha)
        else:
            ax.scatter(xs, ys, zs, marker=marker, alpha=alpha, edgecolor="none")

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    """
    plt.savefig(
        fn,
        bbox_inches='tight',
        pad_inches=0,
        dpi=300, )
    plt.close()
    """
    pass












"""
from sympy import *
x, y, z = symbols("x y z")
"""


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)




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
    o3d.visualization.draw_geometries([line_set, pcd])



def visualize_pts_att_(pts, label_map):
    pcds = []
    C = create_color_samples(label_map.shape[-1])
    for i in range(pts.shape[0]):
        pts_ = np.array(pts[i, ...], dtype=np.float)
        label_map_ = onehot_to_int(label_map[i, ...])
        colors = []
        for i in range(pts_.shape[0]):
            """
            if label_map_[i] == 15:
                colors.append([0., 0., 1.])
            else:
                colors.append([0., 1., 0.])
            """
            colors.append([C[label_map_[i], 0], C[label_map_[i], 1], C[label_map_[i], 2]])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcds.append(pcd)
    o3d.visualization.draw_geometries(pcds)




def tf_center(x):
    c = tf.reduce_mean(x, axis=1, keepdims=True)
    return tf.subtract(x, c)

def tf_random_rotation(shape):
    if isinstance(shape, int):
        shape = [shape]

    batch_size = shape[0]
    t = tf.random.uniform(shape + [3], minval=0., maxval=1.)
    c1 = tf.cos(2 * np.pi * t[:, 0])
    s1 = tf.sin(2 * np.pi * t[:, 0])

    c2 = tf.cos(2 * np.pi * t[:, 1])
    s2 = tf.sin(2 * np.pi * t[:, 1])

    z = tf.zeros(shape)
    o = tf.ones(shape)

    R = tf.stack([c1, s1, z, -s1, c1, z, z, z, o], axis=-1)
    R = tf.reshape(R, shape + [3, 3])

    v1 = tf.sqrt(t[:, -1])
    v3 = tf.sqrt(1-t[:, -1])
    v = tf.stack([c2 * v1, s2 * v1, v3], axis=-1)
    H = tf.tile(tf.expand_dims(tf.eye(3), axis=0), (batch_size, 1, 1)) - 2.* tf.einsum('bi,bj->bij', v, v)
    M = -tf.einsum('bij,bjk->bik', H, R)
    return M

def tf_random_rotate(x):
    R = tf_random_rotation(x.shape[0])
    return tf.einsum('bij,bpj->bpi', R, x)

def var_normalize(x):
    c = tf.reduce_mean(x, axis=1, keepdims=True)
    x = tf.subtract(x, c)
    n2 = tf.multiply(x, x)
    n2 = tf.reduce_sum(n2, axis=-1, keepdims=True)
    n2 = tf.reduce_mean(n2, axis=1, keepdims=True)
    sigma = tf.sqrt(n2 + 0.000000001)
    x = tf.divide(x, sigma)
    return x

def registration(x, y):
    x = var_normalize(x)
    y = var_normalize(y)
    xyt = tf.einsum('bvi,bvj->bij', x, y)
    s, u, v = tf.linalg.svd(xyt)
    R = tf.matmul(v, u, transpose_b=True)
    return R

def lexicographic_ordering(x):
    m = tf.reduce_min(x, axis=1, keepdims=True)
    y = tf.subtract(x, m)
    M = tf.reduce_max(y, axis=1, keepdims=True)
    y = tf.multiply(M[..., 2]*M[..., 1], y[..., 0]) + tf.multiply(M[..., 2], y[..., 1]) + y[..., 2]
    batch_idx = tf.range(y.shape[0])
    batch_idx = tf.expand_dims(batch_idx, axis=-1)
    batch_idx = tf.tile(batch_idx, multiples=(1, y.shape[1]))
    idx = tf.argsort(y, axis=1)
    idx = tf.stack([batch_idx, idx], axis=-1)
    return tf.gather_nd(x, idx)



def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    print(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


NUM_POINTS = 1024

BATCH_SIZE = 32
SHUFFLE = True

def load_dataset(dataset):
    batch_size = BATCH_SIZE
    num_points = NUM_POINTS

    train_files_list = dataset['train_files_list']
    val_files_list = dataset['val_files_list']
    test_files_list = dataset['test_files_list']

    train_data_folder = dataset['train_data_folder']
    val_data_folder = dataset['val_data_folder']
    test_data_folder = dataset['test_data_folder']

    train_preprocessing = dataset['train_preprocessing']
    val_preprocessing = dataset['val_preprocessing']
    test_preprocessing = dataset['test_preprocessing']
    NUM_SAMPLES_DOWNSAMPLED = None
    """
    train_provider = Provider(files_list=train_files_list,
                              data_path=train_data_folder,
                              n_points=num_points,
                              n_samples=NUM_SAMPLES_DOWNSAMPLED,
                              batch_size=batch_size,
                              preprocess=train_preprocessing,
                              shuffle=SHUFFLE)

    val_provider = Provider(files_list=val_files_list,
                            data_path=val_data_folder,
                            n_points=num_points,
                            n_samples=NUM_SAMPLES_DOWNSAMPLED,
                            batch_size=batch_size,
                            preprocess=val_preprocessing,
                            shuffle=SHUFFLE)
    """

    test_provider = Provider(files_list=test_files_list,
                             data_path=test_data_folder,
                             n_points=num_points,
                             n_samples=NUM_SAMPLES_DOWNSAMPLED,
                             batch_size=batch_size,
                             preprocess=test_preprocessing,
                             shuffle=False)

    return test_provider




def save_obj( x, filename: str ):
    """Saves a WavefrontOBJ object to a file

    Warning: Contains no error checking!

    """
    with open( filename, 'w' ) as ofile:
        for i in range(x.shape[0]):
            xi = [x[i, 0], x[i, 1], x[i, 2]]
            ofile.write('v ' + ' '.join(['{}'.format(v) for v in xi]) + '\n')

def tf_random_dir(shape):
    if isinstance(shape, int):
        shape = [shape]
    t = tf.random.uniform(shape + [3], minval=-1., maxval=1.)
    return tf.linalg.l2_normalize(t, axis=-1)

def partial_shapes(x):
    num_points = x.shape[1]
    num_points_partial = num_points // 2
    batch_size = x.shape[0]
    v = tf_random_dir(batch_size)
    vx = tf.einsum('bj,bpj->bp', v, x)
    x_partial_idx = tf.argsort(vx, axis=-1)
    x_partial_idx = x_partial_idx[:, :num_points_partial]
    batch_idx = tf.expand_dims(tf.range(batch_size), axis=1)
    batch_idx = tf.tile(batch_idx, (1, num_points_partial))
    x_partial_idx = tf.stack([batch_idx, x_partial_idx], axis=-1)
    x_partial = tf.gather_nd(x, x_partial_idx)
    x_partial, kd_idx, kd_idx_inv = kdtree_indexing_(x_partial)
    x_partial = tf_center(x_partial)
    return x_partial, x_partial_idx, kd_idx_inv




test_provider = load_dataset(datsets_list[0])



inputs = tf.keras.layers.Input(batch_shape=(BATCH_SIZE, NUM_POINTS // 2, 3))


autoencoder = tf.keras.models.Model(inputs=inputs, outputs=TFN(1024)(inputs), trainable=False)


print(autoencoder.layers)

# weights = 'ckpt-2.data-00000-of-00001'
autoencoder.load_weights('E:/Users/Adrien/Documents/results/tfn_capsules/weights_0.h5')




def save_results(path, data):
    pass

"""
h5_filename = "E:/Users/Adrien/Documents/Datasets/ModelNet40_hdf5/modelnet40_hdf5_1024_classes/data_hdf5/test_toilet.h5"

# h5_filename = "E:/Users/Adrien/Documents/Datasets/ModelNet40_hdf5/modelnet40_hdf5_1024_original/data_hdf5/test_data_0.h5"

f = h5py.File(h5_filename, mode='r')
data = f['data'][:]

# data, labels = load_h5(h5_filename)

print(data.shape)
# print(labels.shape)
"""

# x = data[17, ...]

def grid_of_shapes(x, N):
    y = []
    dx = 2.0
    dy = 2.0
    k = 0
    for i in range(N):
        for j in range(N):
            y.append(tf.stack([x[k, ..., 0] + i*dx, x[k, ..., 1] + j*dy, x[k, ..., 2]], axis=-1))
            k += 1
    return tf.stack(y, axis=0)



batch_idx = 0
shape_idx = 0
X = test_provider.get_data()
print(X.shape)

x = test_provider[batch_idx]

x, _, _ = partial_shapes(x)

x = tf.stack([x[..., 0], x[..., 2], x[..., 1]], axis=-1)

x = tf_center(x)
R = tf_random_rotation(x.shape[0])
x = tf.einsum('bij,bpj->bpi', R, x)

# x = tf_random_rotate(x)
x = kdtree_indexing(x)







caps, z, _ = autoencoder(x, training=False)

z = tf_center(z)

caps_mean = tf.reduce_mean(caps, axis=1, keepdims=True)
# caps = tf.divide(caps, caps_mean + 0.000001)

x = tf.einsum('bji,bpj->bpi', R, x)

# R = registration(x, z)


# z = tf.einsum("bij,bvj->bvi", R, x)
"""
for i in range(BATCH_SIZE):
    zi = np.array(z[i, ...])

    save_obj(zi, 'E:/Users/Adrien/Documents/Datasets/ModelNet40_hdf5/modelnet40_hdf5_1024_classes/tests/test_toilet_chair_' + str(i) + '.obj')
"""

"""
h5_fout = h5py.File("E:/Users/Adrien/Documents/Datasets/ModelNet40_hdf5/modelnet40_hdf5_1024_classes/test_car_chairs.h5")

h5_fout.create_dataset(
        'data', data=np.array(z),
        compression='gzip', compression_opts=4,
        dtype='float32')
"""


i = shape_idx

# z = 0.125*z[i:i+9, ...]
z = z[i:i+9, ...]
x = x[i:i+9, ...]
caps = caps[i:i+9, ...]

# z = 0.125*z[i, ...]
# z = z[i, ...]
# x = x[i, ...]
# caps = caps[i, ...]
# y = y[i, ...]
# print(tf.reduce_max(z))
# print(tf.reduce_max(x))
# x = np.array(x)
# z = np.array(z)
x = grid_of_shapes(x, 3)
z = grid_of_shapes(z, 3)



visualize_pts_att_(z, caps)


"""

print(x.shape)
setup_pcl_viewer(X=x, color=(0.85, 0.85, 1, .5), run=True, point_size=8)

setup_pcl_viewer(X=z, color=(0.85, 1, 0.85, .5), run=True, point_size=8)
vispy.app.run()
"""