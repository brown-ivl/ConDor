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

from auto_encoder.tfn_canonicalizer import TFN
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
SHUFFLE = False

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

    test_provider = Provider(files_list=test_files_list,
                             data_path=test_data_folder,
                             n_points=num_points,
                             n_samples=NUM_SAMPLES_DOWNSAMPLED,
                             batch_size=batch_size,
                             preprocess=test_preprocessing,
                             shuffle=False)

    return train_provider, val_provider, test_provider




def save_obj( x, filename: str ):
    """Saves a WavefrontOBJ object to a file

    Warning: Contains no error checking!

    """
    with open( filename, 'w' ) as ofile:
        for i in range(x.shape[0]):
            xi = [x[i, 0], x[i, 1], x[i, 2]]
            ofile.write('v ' + ' '.join(['{}'.format(v) for v in xi]) + '\n')




train_provider, val_provider, test_provider = load_dataset(datsets_list[0])



inputs_x = tf.keras.layers.Input(batch_shape=(int(BATCH_SIZE/2), NUM_POINTS, 3))
inputs_y = tf.keras.layers.Input(batch_shape=(int(BATCH_SIZE/2), NUM_POINTS, 3))
inputs = [inputs_x, inputs_y]
autoencoder = tf.keras.models.Model(inputs=inputs, outputs=TFN(1024)(inputs), trainable=False)


print(autoencoder.layers)

# weights = 'ckpt-2.data-00000-of-00001'
autoencoder.load_weights('E:/Users/Adrien/Documents/results/pose_and_pts_canonicalization_tfn/weights_0.h5')




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


X = test_provider[1]



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
x = tf.split(X, axis=0, num_or_size_splits=2)

ie_x, ie_y, ied_x, ied_y, it_x, it_y, itd_x, itd_y = autoencoder([x[0], x[1]], training=False)

ie = tf.concat([ie_x, ie_y], axis=0)
it = tf.concat([it_x, it_y], axis=0)
itd = tf.concat([itd_x, itd_y], axis=0)

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


i = 14
ie = 0.125*ie[i, ...]
it = 0.125*it[i, ...]
itd = itd[i, ...]


x = X[i, ...]

x = np.array(x)
ie = np.array(ie)
it = np.array(it)
itd = np.array(itd)


print(x.shape)
setup_pcl_viewer(X=x, color=(0.85, 0.85, 1, .5), run=True, point_size=8)

# setup_pcl_viewer(X=ie, color=(0.85, 1, 0.85, .5), run=True, point_size=8)

setup_pcl_viewer(X=ie, color=(1., 0.85, 0.85, .5), run=True, point_size=8)
vispy.app.run()