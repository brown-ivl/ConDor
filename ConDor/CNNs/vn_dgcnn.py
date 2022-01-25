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
        split_size.append(2 * int(l) + 1)
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
    n2 = tf.reduce_sum(x * x, axis=-1, keepdims=True)
    n2 = tf.expand_dims(n2, axis=-1)
    p = [tf.ones(n2.shape)]
    for m in range(m):
        p.append(p[-1] * n2)

    y = tf_spherical_harmonics(l_max=max_deg).compute(x)
    for l in y:
        y[l] = tf.expand_dims(y[l], axis=-1)

    z = dict()
    for d in range(max_deg + 1):
        z[str(d)] = []
    for l in y:
        l_ = int(l)
        for d in range(m + 1):
            d_ = 2 * d + l_
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
        # y = Activation('relu')(y)
        y = tf.nn.leaky_relu(y)
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

class VNLinear(tf.keras.Model):
    def __init__(self, out_channels, types):
        super(VNLinear, self).__init__()
        self.types = types
        self.map_to_feat = set_interwiners(out_channels, types)
        self.map_to_dir = set_interwiners(out_channels, types)
    def call(self, x):
        # Linear
        return apply_interwiners(x, self.map_to_feat)


class VNLinearLeakyReLU(tf.keras.Model):
    def __init__(self, out_channels, types, momentum, negative_slope=0.2, eps = 1e-6):
        super(VNLinearLeakyReLU, self).__init__()
        self.types = types
        self.negative_slope = negative_slope
        self.map_to_feat = set_interwiners(out_channels, types)
        self.map_to_dir = set_interwiners(out_channels, types)
        self.batchnorm = VNBatchNorm(momentum)
        self.eps = eps
    def call(self, x):
        # Linear
        p = apply_interwiners(x, self.map_to_feat)
        p = concat_types(p)
        # BatchNorm
        p = self.batchnorm(p)
        # LeakyReLU
        d = apply_interwiners(x, self.map_to_dir)
        d = concat_types(d)
        dotprod = tf.reduce_sum(tf.multiply(p, d), axis=-2, keepdims=True)
        mask = tf.cast(dotprod >= 0, dtype=tf.float32)
        d_norm_sq = tf.reduce_sum(tf.multiply(d, d), axis=-2, keepdims=True)
        y = self.negative_slope * p + (1 - self.negative_slope) * (
                    mask * p + (1 - mask) * (p - (dotprod / (d_norm_sq + self.eps)) * d))
        y = split_types(y, self.types)
        return y


class VNBatchNorm(tf.keras.Model):
    def __init__(self, momentum, eps=1e-6):
        super(VNBatchNorm, self).__init__()
        self.eps = eps
        self.bn = BatchNormalization(momentum=momentum)
    def call(self, x):
        norm = tf.reduce_sum(tf.multiply(x, x), axis=-2, keepdims=True)
        norm_bn = self.bn(norm)
        norm_ratio = tf.divide(norm_bn, norm + self.eps)
        return tf.multiply(norm_ratio, x)


class VNStdFeature(tf.keras.Model):
    def __init__(self, in_channels, types, momentum, negative_slope=0.2):
        super(VNStdFeature, self).__init__()


        self.vn1 = VNLinearLeakyReLU(in_channels // 2, types,
                                     momentum=momentum,
                                     negative_slope=negative_slope)
        self.vn2 = VNLinearLeakyReLU(in_channels // 4, types,
                                     momentum=momentum,
                                     negative_slope=negative_slope)
        d = 0
        for l in types:
            d += 2*int(l) + 1
        self.vn_lin = VNLinear(out_channels=d, types=types)

    def call(self, x):
        z0 = x
        z0 = self.vn1(z0)
        z0 = self.vn2(z0)
        z0 = self.vn_lin(z0)
        x = concat_types(x)
        z0 = concat_types(z0)
        x_std = tf.matmul(z0, x, transpose_a=True)
        return x_std, z0


def flatten(x, axis):
    x_shape = list(x.shape)
    axis = axis % len(x_shape)
    new_shape = [-1]
    for i in range(axis, len(x_shape)):
        new_shape.append(x_shape[i])
    y = tf.reshape(x, new_shape)
    return y


"""
def flatten(x, axis):
    axis = axis % len(x.shape)
    axes = list(range(len(x.shape)))
    axes.pop(axis)
    axes.append(axis)
    y = tf.transpose(x, axes)
    y = tf.reshape(y, (-1, x.shape[axis]))
    return y
"""
def channel_first(x):
    axes = list(range(len(x.shape)))
    nc = axes.pop(-1)
    axes.insert(1, nc)
    return tf.transpose(x, axes)

def channel_last(x):
    axes = list(range(len(x.shape)))
    nc = axes.pop(1)
    axes.append(nc)
    return tf.transpose(x, axes)

"""
class VNMaxPool(tf.keras.Model):
    def __init__(self, in_channels, types, axis=2):
        super(VNMaxPool, self).__init__()
        self.types = types
        self.map_to_dir = set_interwiners(in_channels, types)
        self.axis = axis

    def call(self, x):
        d = apply_interwiners(x, self.map_to_dir)
        d = concat_types(d)
        d = tf.stop_gradient(d)
        x = concat_types(x)
        dotprod = tf.reduce_sum(tf.multiply(x, d), axis=-2, keepdims=False)
        x = channel_first(x)
        dotprod = channel_first(dotprod)
        x_shape = list(x.shape)
        x = flatten(x, self.axis+1)
        dotprod = flatten(dotprod, self.axis+1)
        idx = tf.argmax(dotprod, axis=1)
        axis_0_idx = tf.cast(tf.range(x.shape[0]), dtype=tf.int64)


        idx = tf.stack([axis_0_idx, idx], axis=-1)
        y = tf.gather_nd(x, idx)


        x_shape.pop(self.axis+1)

        y = tf.reshape(y, x_shape)
        y = channel_last(y)
        y = split_types(y, self.types)
        return y
"""

class VNMaxPool(tf.keras.Model):
    def __init__(self, in_channels, types, axis=2):
        super(VNMaxPool, self).__init__()
        self.types = types
        self.map_to_dir = set_interwiners(in_channels, types)
        self.axis = axis

    def call(self, x):
        d = apply_interwiners(x, self.map_to_dir)
        d = concat_types(d)
        d = tf.stop_gradient(d)
        x = concat_types(x)
        dotprod = tf.reduce_sum(tf.multiply(x, d), axis=-2, keepdims=False)
        x = channel_first(x)
        dotprod = channel_first(dotprod)
        x_shape = list(x.shape)
        x = flatten(x, self.axis+1)
        dotprod = flatten(dotprod, self.axis+1)
        idx = tf.argmax(dotprod, axis=1)
        axis_0_idx = tf.cast(tf.range(x.shape[0]), dtype=tf.int64)


        idx = tf.stack([axis_0_idx, idx], axis=-1)
        y = tf.gather_nd(x, idx)


        x_shape.pop(self.axis+1)

        y = tf.reshape(y, x_shape)
        y = channel_last(y)
        y = split_types(y, self.types)
        return y


class VNKdMaxPool(tf.keras.Model):
    def __init__(self, in_channels, types, pool_ratio, axis=1):
        super(VNKdMaxPool, self).__init__()
        self.pool_ratio = pool_ratio
        self.axis = axis
        self.types = types
        self.max_pool = VNMaxPool(in_channels, types, axis=axis+1)

    def call(self, x):
        x = concat_types(x)
        shape = list(x.shape)
        shape.insert(self.axis+1, self.pool_ratio)
        shape[self.axis] = -1
        x = tf.reshape(x, shape)
        x = split_types(x, types=self.types)
        y = self.max_pool(x)
        return y

def get_graph_feature(x, patch_size, pool_ratio, spacing, types):
    y = concat_types(x)
    shape = list(y.shape)
    y = tf.reshape(y, (shape[0], shape[1], -1))
    target_points = kd_pooling_1d(y, pool_ratio, pool_mode='avg')
    p_idx, r2 = patches_idx(y, target_points, patch_size, spacing)
    target_points = tf.expand_dims(target_points, axis=2)
    y = tf.gather_nd(y, p_idx)



    y = tf.subtract(y, target_points)
    target_points = tf.tile(target_points, (1, 1, y.shape[2], 1))

    y = tf.concat([tf.subtract(y, target_points), target_points], axis=-1)
    y = tf.reshape(y, (y.shape[0], y.shape[1], y.shape[2], shape[-2], -1))
    y = split_types(y, types)
    return y

"""
def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)
"""
class DGCNN(tf.keras.Model):
    def __init__(self, num_classes):
        super(DGCNN, self).__init__()
        self.num_classes = num_classes
        self.num_points = [1024, 512, 256, 256, 256]
        self.patch_size = [40, 32, 32, 32]

        self.spacing = [0, 0, 0, 0]

        self.units = [4, 64, 64, 128, 256]
        # self.units = [64, 128, 256, 512]

        # self.mlp_units = [[64], [256], [1024]]
        self.bn_momentum = 0.9
        self.droupout_rate = 0.5

        self.l_max = 1
        self.l_list = [str(l) for l in range(self.l_max + 1)]

        self.vn_layers = []
        self.pool_layers = []
        self.global_pool = []
        self.zer_embd_interwiner = set_interwiners(units=self.units[0], types=self.l_list)
        eq_dim = (self.l_max + 1)**2
        for i in range(1, len(self.units)):
            self.vn_layers.append(VNLinearLeakyReLU(out_channels=(self.units[i]//eq_dim),
                                                    types=self.l_list,
                                                    momentum=self.bn_momentum))
            """
            self.pool_layers.append(VNKdMaxPool(in_channels=(self.units[i]//eq_dim),
                                                types=self.l_list,
                                                pool_ratio=int(self.num_points[i-1] / self.num_points[i])))
            """
            self.pool_layers.append(VNMaxPool(in_channels=(self.units[i]//eq_dim), axis=2, types=self.l_list))

            """
            self.global_pool.append(VNKdMaxPool(in_channels=(self.units[i] // eq_dim), axis=2, types=self.l_list,
                                                pool_ratio=int(self.num_points[i-1] / self.num_points[-1])))
            """
        self.vn_layers.append(VNLinearLeakyReLU(1024 // eq_dim,
                                                types=self.l_list,
                                                momentum=self.bn_momentum))

        self.vn_std = VNStdFeature(2*1024 // eq_dim, types=self.l_list, momentum=self.bn_momentum)

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
        # y = concat_types(y)


        Y = []
        for i in range(1, len(self.units)):
            y = get_graph_feature(y,
                              patch_size=self.patch_size[i-1],
                              spacing=0,
                              pool_ratio=int(self.num_points[i - 1] / self.num_points[i]),
                              types=self.l_list)
            y = self.vn_layers[i-1](y)
            # y = concat_types(y)
            # y = tf.reduce_mean(y, axis=2, keepdims=False)
            # y = split_types(y, self.l_list)
            y = self.pool_layers[i-1](y)
            y_ = concat_types(y)
            print(y_.shape)
            y_ = kd_pooling_2d(y_, pool_size=int(self.num_points[i] / self.num_points[-1]), pool_mode='avg')
            Y.append(y_)

        y = tf.concat(Y, axis=-1)
        y = split_types(y, self.l_list)
        y = self.vn_layers[-1](y)
        y = concat_types(y)
        y_mean = tf.reduce_mean(y, axis=1, keepdims=True)
        y_mean = tf.tile(y_mean, (1, y.shape[1], 1, 1))
        y = tf.concat([y, y_mean], axis=-1)
        y = split_types(y, self.l_list)
        y, _ = self.vn_std(y)
        y = tf.reshape(y, (y.shape[0], y.shape[1], -1))
        y1 = tf.reduce_max(y, axis=1, keepdims=False)
        y2 = tf.reduce_mean(y, axis=1, keepdims=False)
        y = tf.concat([y1, y2], axis=-1)

        # y = apply_mlp(y, self.mlp)

        # y = tf.reduce_max(y, axis=1, keepdims=False)
        y = self.fc1(y)
        y = self.bn_fc1(y)
        # y = Activation('relu')(y)
        y = tf.nn.leaky_relu(y)
        y = Dropout(rate=self.droupout_rate)(y)
        y = self.fc2(y)
        y = self.bn_fc2(y)
        # y = Activation('relu')(y)
        y = tf.nn.leaky_relu(y)
        y = Dropout(rate=self.droupout_rate)(y)
        y = self.softmax(y)
        return y