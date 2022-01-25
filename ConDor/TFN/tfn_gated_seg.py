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
from spherical_harmonics.kernels import ZernikeGaussianKernelConv, ZernikeGaussianKernels, \
    SphericalHarmonicsGaussianKernels, ZernikePolynomials
from activations import DodecahedronEval, DodecahedronCoeffs
from group_points import GroupPoints
from pooling import kd_pooling_2d, kd_pooling_1d
from spherical_harmonics.kernels import tf_monomial_basis_3D_idx, tf_spherical_harmonics_basis, tf_eval_monom_basis
import numpy as np
from pooling import aligned_kdtree_indexing_, kdtree_indexing_
from utils.pointclouds_utils import tf_kd_tree_idx
from activations import apply_norm_layer, set_norm_activation_layers_sphere
from activations import apply_gated_layer, set_gater_layer

def set_mlp(units, momentum):
    mlp = []
    for i in range(len(units)):
        layer = Dense(units=units[i])
        bn_layer = BatchNormalization(momentum=momentum)
        mlp.append({'layer': layer, 'bn_layer': bn_layer})
    return mlp


def apply_mlp(x, mlp, axes, mode='or', split_ratio=None):
    y = x
    for i in range(len(mlp)):
        y = mlp[i]['layer'](y)
        y = mlp[i]['bn_layer'](y)
        y = Activation('relu')(y)
    return y


def set_sphere_weights(units, types):
    weights = dict()
    for l in types:
        if int(l) == 0:
            weights[l] = Dense(units=units)
        else:
            weights[l] = Dense(units=units, use_bias=False)
    return weights


def apply_layers(x, layers):
    y = dict()
    for l in x:
        if l.isnumeric():
            y[l] = layers[l](x[l])
    return y


def norms(x):
    y = []
    for l in x:
        if l.isnumeric():
            nxl = tf.reduce_sum(tf.multiply(x[l], x[l]), axis=-2, keepdims=False)
            y.append(nxl)
    n = tf.concat(y, axis=-1)
    n = tf.sqrt(tf.maximum(n, 1e-8))
    return n

def feature_propagation(x, idx, dist, eps=1e-8):
    y = tf.gather_nd(x, idx)
    sq_dist = tf.multiply(dist, dist)
    w = tf.math.reciprocal(sq_dist + eps)
    w = tf.divide(w, tf.reduce_sum(w, axis=-1, keepdims=True))
    w_shape = list(w.shape)
    for i in range(len(y.shape) - len(w_shape)):
        w_shape.append(1)
    w = tf.reshape(w, w_shape)
    y = tf.multiply(w, y)
    return tf.reduce_sum(y, axis=2, keepdims=False)

def split_sphere_signal(x, l_max):
    y = dict()
    split_size = []
    for l in range(l_max+1):
        split_size.append(2*l+1)
    y_ = tf.split(x, num_or_size_splits=split_size, axis=-2)
    for l in range(l_max+1):
        y[str(l)] = y_[l]
    return y

def stack_sphere_signal(x):
    l_max = 0
    for l in x:
        if l.isnumeric():
            l_max = max(l_max, int(l))
    y = []
    for l in range(l_max+1):
        y.append(x[str(l)])
    return tf.concat(y, axis=-2)

class SegNetNorm(tf.keras.Model):
    def __init__(self, num_classes):
        super(SegNetNorm, self).__init__()
        self.num_classes = num_classes
        self.dodecahedron = 'pentakis'
        # self.dodecahedron = 'regular'
        self.d = 3
        self.l_max = [3, 3, 3]
        self.l_max_out = 3
        self.l_max_coeffs = [3, 3, 3]
        self.num_shells = [3, 3, 3]
        self.gaussian_scale = []

        for i in range(len(self.num_shells)):
            self.gaussian_scale.append(0.69314718056 * ((self.num_shells[i]) ** 2))
        self.radius = [0.2, 0.4, 0.8]
        self.bounded = [True, True, True]
        self.num_points = [2048, 512, 128, 32]
        self.patch_size = [64, 64, 64]
        self.spacing = [0, 0, 0]

        self.equivariant_units = [16, 32, 64]

        self.equivariant_units_up = [64, 32, 32]

        self.mlp_units = [[16], [32], [64]]
        self.mlp_units_up = [[64], [32], [32]]

        self.bn_momentum = 0.75
        self.patches_norm_momentum = 0.95
        self.droupout_rate = 0.2

        self.grouping_layers = []
        self.kernel_layers = []
        self.conv_layers = []

        for i in range(len(self.radius)):
            gi = GroupPoints(radius=self.radius[i],
                             patch_size_source=self.patch_size[i],
                             spacing_source=self.spacing[i],
                             radius_target=1.0,
                             patch_size_target=3)
            self.grouping_layers.append(gi)




            ki = SphericalHarmonicsGaussianKernels(l_max=self.l_max[i],
                                                   gaussian_scale=self.gaussian_scale[i],
                                                   num_shells=self.num_shells[i],
                                                   bound=self.bounded[i])
            ci = ShGaussianKernelConv(l_max=self.l_max[i], l_max_out=self.l_max_out)


            self.kernel_layers.append(ki)
            self.conv_layers.append(ci)


        self.mlp = []
        self.equivariant_weights = []
        self.bn = []

        """
        self.zernike_layer = []
        for i in range(len(self.radius)):
            self.zernike_layer.append(set_zernike_weights(units=self.equivariant_units[i], d=self.l_max_coeffs[i]))
        """

        for i in range(len(self.radius)):
            self.bn.append(BatchNormalization(momentum=self.bn_momentum))
            types = [str(l) for l in range(self.l_max_out + 1)]
            self.equivariant_weights.append(set_gater_layer(units=self.equivariant_units[i],
                                                            l_max=self.l_max_out,
                                                            momentum=self.bn_momentum))
            self.mlp.append(set_mlp(self.mlp_units[i], self.bn_momentum))




        self.mlp_up = []
        self.equivariant_weights_up = []
        self.bn_up = []
        for i in range(len(self.radius)):
            self.bn_up.append(BatchNormalization(momentum=self.bn_momentum))
            types = [str(l) for l in range(self.l_max_out + 1)]
            self.mlp_up.append(set_mlp(self.mlp_units[i], self.bn_momentum))
            self.equivariant_weights_up.append(set_gater_layer(units=self.equivariant_units_up[i],
                                                               l_max=self.l_max_out,
                                                               momentum=self.bn_momentum))


        self.fc1_units = 128
        self.fc2_units = 128

        self.fc1 = Dense(units=self.fc1_units, activation=None)
        # if with_bn:
        self.bn_fc1 = BatchNormalization(momentum=self.bn_momentum)
        # self.activation1 = Activation('relu')
        # self.drop1 = Dropout(rate=self.droupout_rate)
        self.fc2 = Dense(units=self.fc2_units, activation=None)
        # if with_bn:
        self.bn_fc2 = BatchNormalization(momentum=self.bn_momentum)
        # self.activation1 = Activation('relu')
        # self.drop2 = Dropout(rate=self.droupout_rate)
        self.softmax = Dense(units=self.num_classes, activation='softmax')

    def call(self, x):
        pts_input, _, kd_idx_inv = kdtree_indexing_(x[0])
        # pts_input = x[0]

        points0 = kd_pooling_1d(pts_input, int(pts_input.shape[1] / self.num_points[0]))
        points = [points0]

        shape_class = x[1]

        grouped_points = []
        kernels = []

        for i in range(len(self.radius)):
            pi = kd_pooling_1d(points[-1], int(self.num_points[i] / self.num_points[i + 1]))
            # pi = Jitter(self.jitter_scale[i])(pi)
            points.append(pi)

        yzx = []
        for i in range(len(points)):
            yzx_i = tf.stack([points[i][..., 1], points[i][..., 2], points[i][..., 0]], axis=-1)
            yzx.append(tf.expand_dims(yzx_i, axis=-1))

        for i in range(len(self.radius)):
            gi = self.grouping_layers[i]({"source points": points[i], "target points": points[i + 1]})
            ki = self.kernel_layers[i]({"patches": gi["patches source"], "patches dist": gi["patches dist source"]})
            grouped_points.append(gi)
            kernels.append(ki)

        Y = []
        y = {'0': tf.ones((points[0].shape[0], points[0].shape[1], 1, 1))}
        for i in range(len(self.radius)):
            y["source points"] = points[i]
            y["target points"] = points[i + 1]
            y["patches idx"] = grouped_points[i]["patches idx source"]
            y["patches dist source"] = grouped_points[i]["patches dist source"]
            y["kernels"] = kernels[i]

            if '1' in y:
                y['1'] = tf.concat([y['1'], yzx[i]], axis=-1)
            else:
                y['1'] = yzx[i]

            y = self.conv_layers[i](y)

            if '1' in y:
                y['1'] = tf.concat([y['1'], yzx[i + 1]], axis=-1)
            else:
                y['1'] = yzx[i + 1]

            y = apply_gated_layer(y, self.equivariant_weights[i])

            Y.append(stack_sphere_signal(y))

        Y.reverse()
        Y.append(yzx)
        grouped_points.reverse()
        y = Y[0]
        for i in range(len(self.radius)-1):
            y = feature_propagation(y,
                                    idx=grouped_points[i]["patches idx target"],
                                    dist=grouped_points[i]["patches dist target"])
            y = tf.concat([y, Y[i+1]], axis=-1)
            y = split_sphere_signal(y, l_max=self.l_max_out)
            y = apply_gated_layer(y, self.equivariant_weights_up[i])
            y = stack_sphere_signal(y)

        y = feature_propagation(y, idx=grouped_points[-1]["patches idx target"],
                                   dist=grouped_points[-1]["patches dist target"])
        y = split_sphere_signal(y, l_max=self.l_max_out)
        if '1' in y:
            y['1'] = tf.concat([y['1'], yzx[0]], axis=-1)
        else:
            y['1'] = yzx[0]


        y = apply_gated_layer(y, self.equivariant_weights_up[-1])
        y = norms(y)

        # y = y_shape
        shape_class = tf.expand_dims(shape_class, axis=1)
        shape_class = tf.tile(shape_class, (1, y.shape[1], 1))
        y = tf.concat([y, shape_class], axis=-1)

        print('last y shape ', y.shape)
        y = self.fc1(y)
        y = self.bn_fc1(y)
        y = Activation('relu')(y)
        # y = Dropout(rate=self.droupout_rate)(y)


        y = self.fc2(y)
        y = self.bn_fc2(y)
        y = Activation('relu')(y)
        y = Dropout(rate=self.droupout_rate)(y)


        y = self.softmax(y)
        y = tf.gather_nd(y, kd_idx_inv)
        # y = tf.reduce_mean(y, axis=1, keepdims=False)
        return y