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
from spherical_harmonics.kernels import ZernikeGaussianKernelConv, ZernikeGaussianKernels, SphericalHarmonicsGaussianKernels
from activations import DodecahedronEval, DodecahedronCoeffs
from group_points import GroupPoints
from pooling import kd_pooling_2d, kd_pooling_1d, kdtree_indexing, aligned_kdtree_indexing
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


class TFN(tf.keras.Model):
    def __init__(self, num_classes):
        super(TFN, self).__init__()
        self.num_classes = num_classes
        self.dodecahedron = 'pentakis'
        # self.dodecahedron = 'regular'
        self.d = 3
        self.l_max = [3, 3, 3]
        self.l_max_out = [3, 3, 3]
        self.num_shells = [3, 3, 3]
        self.gaussian_scale = []
        for i in range(len(self.num_shells)):
            self.gaussian_scale.append(0.69314718056 * ((self.num_shells[i]) ** 2))
        self.radius = [0.2, 0.40, 0.8]
        self.bounded = [True, True, True]
        self.num_points = [1024, 256, 64, 16]
        self.patch_size = [32, 32, 32]
        self.spacing = [0, 0, 0]
        self.equivariant_units = [32, 64, 128]
        self.mlp_units = [[32, 32], [64, 64], [128, 256]]
        self.bn_momentum = 0.75
        self.droupout_rate = 0.5

        self.grouping_layers = []
        self.kernel_layers = []
        self.conv_layers = []
        self.eval = []
        self.coeffs = []

        for i in range(len(self.radius)):
            gi = GroupPoints(radius=self.radius[i],
                             patch_size_source=self.patch_size[i],
                             spacing_source=self.spacing[i])
            self.grouping_layers.append(gi)

            ki = SphericalHarmonicsGaussianKernels(l_max=self.l_max[i],
                                                   gaussian_scale=self.gaussian_scale[i],
                                                   num_shells=self.num_shells[i],
                                                   bound=self.bounded[i])
            ci = ShGaussianKernelConv(l_max=self.l_max[i], l_max_out=self.l_max_out[i])

            self.kernel_layers.append(ki)
            self.conv_layers.append(ci)

            # self.eval.append(DodecahedronEval(l_max=self.l_max_out[i], dodecahedron=self.dodecahedron))
            # self.coeffs.append(DodecahedronCoeffs(l_max=self.l_max_out[i], dodecahedron=self.dodecahedron))

        self.equivariant_weights = []
        self.bn = []

        for i in range(len(self.radius)):
            self.bn.append(BatchNormalization(momentum=self.bn_momentum))
            types = [str(l) for l in range(self.l_max_out[i] + 1)]
            self.equivariant_weights.append(set_sphere_weights(self.equivariant_units[i], types=types))


        self.mlp_sphere = set_mlp(units=[128, 256, 512], momentum=self.bn_momentum)
        self.mlp_shape = set_mlp(units=[128, 256, 512], momentum=self.bn_momentum)

        self.fc1_units = 512
        self.fc2_units = 256

        self.fc1 = Dense(units=self.fc1_units, activation=None)
        self.bn_fc1 = BatchNormalization(momentum=self.bn_momentum)
        self.fc2 = Dense(units=self.fc2_units, activation=None)
        self.bn_fc2 = BatchNormalization(momentum=self.bn_momentum)
        self.softmax = Dense(units=self.num_classes, activation='softmax')

    def call(self, x):

        # x = kdtree_indexing(x)
        x = aligned_kdtree_indexing(x)

        points = [x]
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

        y = {'0': tf.ones((x.shape[0], x.shape[1], 1, 1))}
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

            y = apply_layers(y, self.equivariant_weights[i])
            y = DodecahedronEval(l_max=self.l_max_out[i], dodecahedron=self.dodecahedron)(y)
            y = self.bn[i](y)
            y = Activation('relu')(y)
            if i < len(self.radius) - 1:
                y = DodecahedronCoeffs(l_max=self.l_max_out[i], dodecahedron=self.dodecahedron)(y)

        y = tf.reduce_max(y, axis=1, keepdims=False)
        y = DodecahedronCoeffs(l_max=self.l_max_out[-1], dodecahedron=self.dodecahedron)(y)
        y = norms(y)

        print('last y shape ', y.shape)
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





