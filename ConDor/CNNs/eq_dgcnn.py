import tensorflow as tf
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout
from spherical_harmonics.kernels import ShGaussianKernelConv
from spherical_harmonics.kernels import SphericalHarmonicsGaussianKernels
from group_points import GroupPoints
from pooling import kd_pooling_2d, kd_pooling_1d, kdtree_indexing, aligned_kdtree_indexing
from activations import apply_norm_layer, set_norm_activation_layers_sphere
from spherical_harmonics.kernels import tf_monomial_basis_3D_idx, tf_spherical_harmonics_basis, tf_eval_monom_basis
import numpy as np
from utils.pointclouds_utils import tf_kd_tree_idx
from SO3_CNN.spherical_harmonics_ import tf_spherical_harmonics

def concat_types(x):
    Y = []
    for l in x:
        if l.isnumeric():
            Y.append(x[l])
    y = tf.concat(Y, axis=-2)
    return y

def split_types(x, types):
    split_size = []
    for l in types:
        split_size.append(2*int(l)+1)
    y_ = tf.split(x, num_or_size_splits=split_size, axis=-2)
    y = dict()
    for i in range(len(types)):
        y[types[i]] = y_[i]
    return y

def set_interwiners(units, types):
    weights = dict()
    for l in types:
        if int(l) == 0:
            weights[l] = Dense(units=units)
        else:
            weights[l] = Dense(units=units, use_bias=False)
    return weights

def apply_interwiners(x, layers):
    y = dict()
    for l in x:
        if l.isnumeric():
            y[l] = layers[l](x[l])
    return y

def zernike_monoms(x, max_deg):
    m = int(max_deg / 2.)
    n2 = tf.reduce_sum(x*x, axis=-1, keepdims=True)
    n2 = tf.expand_dims(n2, axis=-1)
    p = [tf.ones(n2.shape)]
    for m in range(m):
        p.append(p[-1]*n2)

    y = tf_spherical_harmonics(l_max=max_deg).compute(x)
    for l in y:
        y[l] = tf.expand_dims(y[l], axis=-1)

    z = dict()
    for d in range(max_deg+1):
        z[str(d)] = []
    for l in y:
        l_ = int(l)
        for d in range(m+1):
            d_ = 2*d + l_
            if d_ <= max_deg:
                print(p[d].shape)
                print(y[l].shape)
                zd = tf.multiply(p[d], y[l])
                z[str(d_)].append(zd)
    for d in z:
        z[d] = tf.concat(z[d], axis=-1)
    return z

def set_mlp(units, momentum):
    mlp = []
    for i in range(len(units)):
        layer = Dense(units=units[i])
        bn_layer = BatchNormalization(momentum=momentum)
        mlp.append({'layer': layer, 'bn_layer': bn_layer})
    return mlp

def apply_mlp(x, mlp):
    y = x
    for i in range(len(mlp)):
        y = mlp[i]['layer'](y)
        y = mlp[i]['bn_layer'](y)
        y = Activation('relu')(y)
    return y

def patches_idx(source, target, num_samples, spacing):
    batch_size = source.shape[0]

    num_points_target = target.shape[1]

    r0 = tf.multiply(target, target)
    r0 = tf.reduce_sum(r0, axis=2, keepdims=True)
    r1 = tf.multiply(source, source)
    r1 = tf.reduce_sum(r1, axis=2, keepdims=True)
    r1 = tf.transpose(r1, [0, 2, 1])
    sq_distance_mat = r0 - 2. * tf.matmul(target, tf.transpose(source, [0, 2, 1])) + r1
    num_points_source = source.shape[1]
    assert (num_samples * (spacing + 1) <= num_points_source)
    sq_patches_dist, patches_idx = tf.nn.top_k(-sq_distance_mat, k=num_samples * (spacing + 1))
    if spacing > 0:
        patches_idx = patches_idx[:, :, 0::(spacing + 1), ...]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    batch_idx = tf.tile(batch_idx, (1, num_points_target, num_samples))
    patches_idx = tf.stack([batch_idx, patches_idx], -1)

    return patches_idx, -sq_patches_dist

class DGCNN(tf.keras.Model):
    def __init__(self, num_classes):
        super(DGCNN, self).__init__()
        self.num_classes = num_classes
        self.num_points = [1024, 512, 256, 256, 256]
        self.patch_size = [40, 32, 32, 32]

        self.spacing = [0, 0, 0, 0]



        self.units = [64, 64, 128, 256]
        # self.units = [64, 128, 256, 512]

        # self.mlp_units = [[64], [256], [1024]]
        self.bn_momentum = 0.9
        self.droupout_rate = 0.5

        self.l_max = 3
        self.l_list = [str(l) for l in range(self.l_max+1)]

        self.bn = []
        self.weights_rel = []
        self.weights_abs = []
        self.interwiners_local = []
        self.interwiners_global = []
        self.interwiners_activation = []

        self.zer_embd_interwiner = set_interwiners(units=4, types=self.l_list)

        for i in range(len(self.units)):
            self.bn.append(BatchNormalization(momentum=self.bn_momentum))
            self.weights_rel.append(Dense(units=self.units[i]))
            self.weights_abs.append(Dense(units=self.units[i]))
            # self.interwiners_local.append(set_interwiners(units=self.units[i], types=['0', '1', '2', '3']))
            # self.interwiners_global.append(set_interwiners(units=self.units[i], types=['0', '1', '2', '3']))
            self.interwiners_activation.append(set_interwiners(units=(self.l_max+1)**2, types=self.l_list))
            # self.mlp.append(set_mlp(self.mlp_units[i], self.bn_momentum))

        self.mlp = set_mlp([1024], momentum=self.bn_momentum)
        self.fc1_units = 512
        self.fc2_units = 256

        self.fc1 = Dense(units=self.fc1_units, activation=None)
        self.bn_fc1 = BatchNormalization(momentum=self.bn_momentum)
        self.fc2 = Dense(units=self.fc2_units, activation=None)
        self.bn_fc2 = BatchNormalization(momentum=self.bn_momentum)
        self.softmax = Dense(units=self.num_classes, activation='softmax')


    def call(self, x):

        x = kdtree_indexing(x)
        # x = aligned_kdtree_indexing(x)
        y = zernike_monoms(x, self.l_max)
        y = apply_interwiners(y, self.zer_embd_interwiner)
        y = concat_types(y)

        """
        target_points = kd_pooling_1d(y, int(self.num_points[0] / self.num_points[1]), pool_mode='avg')
        p_idx, r2 = patches_idx(y, target_points, self.patch_size[i], self.spacing[i])
        r = tf.sqrt(r2 + 0.000000001)
        y_patches = tf.gather_nd(y, p_idx)
        r_mean = tf.reduce_mean(r, axis=1, keepdims=True)
        target_points = tf.expand_dims(target_points, axis=2)
        patches_rel = tf.subtract(y_patches, target_points)
        patches_rel = 0.75*tf.divide(patches_rel, r_mean)
        local_zernike = zernike_monoms(patches_rel, 3)
        y_global = apply_interwiners(global_zernike, self.interwiners_global[0])
        y_local = apply_interwiners(local_zernike, self.interwiners_local[0])
        y_local = tf.expand_dims(y_local, axis=2)
        y = tf.add(y_local, y_global)
        y = concat_types(y)
        basis = tf.reduce_mean(y, axis=2, keepdims=True)
        basis = tf.linalg.l2_normalize(basis, axis=-2, epsilon=0.001)
        # y = tf.einsum('bvij,bvpic->bvpjc', basis, y)
        y = tf.matmul(basis, y, transpose_a=True)
        y = Activation('relu')(y)
        # y = tf.einsum('bvij,bvpjc->bvpic', basis, y)
        y = tf.matmul(basis, y)
        """

        Y = []
        for i in range(1, len(self.units)):
            shape = list(y.shape)
            y = tf.reshape(y, (shape[0], shape[1], -1))
            target_points = kd_pooling_1d(y, int(self.num_points[i] / self.num_points[i+1]), pool_mode='avg')
            p_idx, r2 = patches_idx(y, target_points, self.patch_size[i], self.spacing[i])
            y = tf.reshape(y, shape)
            target_points = tf.reshape(target_points, (shape[0], -1, shape[2], shape[3]))
            y_patches = tf.gather_nd(y, p_idx)
            y_patches_mean = tf.reduce_mean(y_patches, axis=2, keepdims=False)
            basis = tf.concat([y_patches_mean, target_points], axis=-1)
            basis = split_types(basis, self.l_list)
            basis = apply_interwiners(basis, self.interwiners_activation[i])
            basis = concat_types(basis)
            basis = tf.linalg.l2_normalize(basis, axis=-2, epsilon=0.001)
            # y = tf.einsum('bvij,bvpic->bvpjc', basis, y)

            y_abs = tf.expand_dims(target_points, axis=2)
            y_rel = tf.subtract(y_patches, y_abs)
            basis = tf.expand_dims(basis, axis=2)
            y_abs = tf.matmul(basis, y_abs, transpose_a=True)
            y_rel = tf.matmul(basis, y_rel, transpose_a=True)
            y_abs = tf.reshape(y_abs, (y_abs.shape[0], y_abs.shape[1], y_abs.shape[2], -1))
            y_rel = tf.reshape(y_rel, (y_rel.shape[0], y_rel.shape[1], y_rel.shape[2], -1))
            y_abs = self.weights_abs[i](y_abs)
            y_rel = self.weights_rel[i](y_rel)
            """
            y_rel = tf.reshape(y_rel, (y_rel_shape[0], y_rel_shape[1], -1))
            
            y_rel_c = self.weights_rel[i](y_abs)
            y_abs_shape = list(y_abs.shapes)
            y_abs = tf.reshape(y_local, (y_abs_shape[0], y_abs_shape[1], -1))
            y_abs = self.weights_abs[i](y_abs)
            
            y_rel = tf.subtract
            
            y_rel = tf.matmul(basis, y, transpose_a=True)
            y_rel_shape = list(y.shape)
            y_rel = tf.reshape(y_rel, (y_rel_shape[0], y_rel_shape[1], -1))
            y_rel = self.weights_rel[i](y_rel)
            # y_rel = self.bn_rel[i](y_rel)

            # y_abs = tf.expand_dims(target_points, axis=2)


            y_abs = tf.expand_dims(y_abs, axis=2)
            y = tf.subtract(y_abs, y_rel_c)
            y = tf.expand_dims(y, axis=2)
            y_rel = tf.gather_nd(y_rel, p_idx)
            """
            y = tf.add(y_abs, y_rel)
            y = self.bn[i](y)
            y = Activation('relu')(y)
            y = tf.reduce_max(y, axis=2, keepdims=False)
            y_ = kd_pooling_1d(y, int(self.num_points[i + 1] / self.num_points[-1]), pool_mode='max')

            y = tf.reshape(y, (y.shape[0], y.shape[1], (self.l_max + 1)**2, -1))
            # y = tf.einsum('bvij,bvpjc->bvpic', basis, y)


            y = tf.matmul(basis[:, :, 0, :, :], y)
            Y.append(y_)

        print(Y)
        y = tf.concat(Y, axis=-1)
        y = apply_mlp(y, self.mlp)

        y = tf.reduce_max(y, axis=1, keepdims=False)
        y = self.fc1(y)
        y = self.bn_fc1(y)
        y = Activation('relu')(y)
        y = Dropout(rate=self.droupout_rate)(y)
        y = self.fc2(y)
        y = self.bn_fc2(y)
        y = Activation('relu')(y)
        y = Dropout(rate=self.droupout_rate)(y)
        y = self.softmax(y)
        return y