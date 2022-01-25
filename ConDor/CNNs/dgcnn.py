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

    return patches_idx

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




        self.bn = []
        self.weights_rel = []
        self.weights_abs = []

        for i in range(len(self.units)):
            self.bn.append(BatchNormalization(momentum=self.bn_momentum))
            self.weights_rel.append(Dense(units=self.units[i]))
            self.weights_abs.append(Dense(units=self.units[i]))
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

        y = x
        Y = []
        for i in range(len(self.units)):
            target_points = kd_pooling_1d(y, int(self.num_points[i] / self.num_points[i + 1]), pool_mode='avg')
            p_idx = patches_idx(y, target_points, self.patch_size[i], self.spacing[i])
            y_patches = tf.gather_nd(y, p_idx)
            target_points = tf.expand_dims(target_points, axis=2)
            patches_rel = tf.subtract(y_patches, target_points)
            patches_rel = self.weights_rel[i](patches_rel)
            patches_abs = self.weights_abs[i](target_points)
            y = tf.add(patches_rel, patches_abs)
            y = self.bn[i](y)
            y = Activation('relu')(y)
            y = tf.reduce_max(y, axis=2, keepdims=False)
            y_ = kd_pooling_1d(y, int(self.num_points[i+1] / self.num_points[-1]), pool_mode='max')
            Y.append(y_)
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