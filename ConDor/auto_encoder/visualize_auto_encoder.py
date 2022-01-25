import tensorflow as tf
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

from auto_encoder.tfn_auto_encoder import TFN
from plyfile import PlyData, PlyElement





import h5py
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

print(tf.__version__)
import numpy as np
from utils.data_prep_utils import load_h5_data
from pooling import GridBatchSampler
from utils.pointclouds_utils import setup_pcl_viewer
import vispy
from activations import tf_dodecahedron_sph
from spherical_harmonics.kernels import monomial_basis_3D
from pooling import simple_grid_sampler

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




test_provider = load_dataset(datsets_list[0])



inputs = tf.keras.layers.Input(batch_shape=(BATCH_SIZE, NUM_POINTS, 3))


autoencoder = tf.keras.models.Model(inputs=inputs, outputs=TFN(1024)(inputs), trainable=False)


print(autoencoder.layers)

# weights = 'ckpt-2.data-00000-of-00001'
autoencoder.load_weights('E:/Users/Adrien/Documents/results/pose_canonicalization_tfn/shapenet_0/weights_0.h5')




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

batch_idx = 0
shape_idx = 9
X = test_provider.get_data()
print(X.shape)

x = test_provider[batch_idx]
x = tf_random_rotate(x)
x = kdtree_indexing(x)



# idx = tf.random.shuffle(tf.range(BATCH_SIZE))

# x = tf.gather(x, idx, axis=0)

"""
x = tf.expand_dims(x[0, ...], axis=0)
x = tf.tile(x, (BATCH_SIZE, 1, 1))
"""

# x = data[164, ...]

# x = np.array(x)
# x = tf.expand_dims(x, axis=0)

"""
y = kdtree_indexing(x, depth=4)
y = kd_pooling_1d(y, int(NUM_POINTS / NUM_POINTS_OUT))
y = lexicographic_ordering(y)
"""






"""
patches = tf_patches(y, y, patch_size=64)

x = np.array(patches[0, 177, ...])
print(x.shape)
"""

# z = autoencoder.predict([x, y], batch_size=BATCH_SIZE)



_, z, _ = autoencoder(x, training=False)

# R = registration(x, z)
x = tf_center(x)

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
z = 0.125*z[i, ...]
# z = z[i, ...]
x = x[i, ...]

# y = y[i, ...]
# print(tf.reduce_max(z))
# print(tf.reduce_max(x))
x = np.array(x)
z = np.array(z)



print(x.shape)
setup_pcl_viewer(X=x, color=(0.85, 0.85, 1, .5), run=True, point_size=8)

setup_pcl_viewer(X=z, color=(0.85, 1, 0.85, .5), run=True, point_size=8)
vispy.app.run()


