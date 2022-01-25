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
# from activations import DodecahedronEval, DodecahedronCoeffs
from group_points import GroupPoints
from pooling import kd_pooling_2d, kd_pooling_1d, kdtree_indexing, aligned_kdtree_indexing
from spherical_harmonics.kernels import tf_monomial_basis_3D_idx, tf_spherical_harmonics_basis, tf_eval_monom_basis
import numpy as np
from utils.pointclouds_utils import tf_kd_tree_idx

from SO3_CNN.spherical_harmonics_ import SphericalHarmonicsCoeffs, SphericalHarmonicsEval
from SO3_CNN.sampling import tf_S2_fps

from circle_bundle_net.kernels import convert_fourier_to_sph, convert_sph_to_fourier
from circle_bundle_net.kernels import SphTensorToFourierIrreps, rotate_real_sph_signal, fourier_relu, LocallyAlignedConv
from circle_bundle_net.kernels import inverse_sph_fourier_transform, sph_fourier_transform
from SO3_CNN.tf_wigner import tf_wigner_matrix
from circle_bundle_net.local_directions import radial_frames, local_direction, frame_from_vector, center_orientation
import math

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

def set_intertwiners(units, types):
    weights = dict()

    if isinstance(units, dict):
        for l in types:
            if int(l) == 0:
                weights[l] = Dense(units=units[l])
            else:
                weights[l] = Dense(units=units[l], use_bias=False)
        return weights

    for l in types:
        if int(l) == 0:
            weights[l] = Dense(units=units)
        else:
            weights[l] = Dense(units=units, use_bias=False)
    return weights


def apply_intertwiners(x, layers):
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

def tf_fibonnacci_sphere_sampling(num_pts):
    indices = np.arange(0, num_pts, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/num_pts)
    theta = np.pi * (1 + 5**0.5) * indices
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    S2 = np.stack([x, y, z], axis=-1)
    return tf.convert_to_tensor(S2, dtype=tf.float32)


def stack_equivariant_features(x):
    types = []
    for l in x:
        if l.isnumeric():
            types.append(int(l))
    types.sort()
    Y = []
    for i in range(len(types)):
        Y.append(x[str(types[i])])
    y = tf.concat(Y, axis=-2)
    return y, types

"""
def apply_gated_layer(x, w):
    l_max = int(x.shape[-2] / 2)
    y0 = w['weights']['0'](x['0'])
    y0 = w['bn']['0'](y0)
    y0 = tf.nn.relu(y0)
    y = dict()
    y[0] = y0
    for l in range(-l_max,l_max+1):
        if l != 0
            k = str(l)
            gl = w['gate_weights'][k](x['0'])
            gl = w['gate_bn'][k](gl)
            gl = tf.sigmoid(gl)
            yl = w['weights'][k](x[k])
            yl = tf.multiply(gl, yl)
            y[l] = yl

    Y = []
    for l in range(-l_max, l_max + 1):
        Y.append(y[l])

    return tf.stack(Y, axis=-2)


def set_gated_layer(units, l_max, momentum):
    w = dict()
    w['bn'] = dict()
    w['weights'] = dict()
    w['gate_weights'] = dict()
    w['gate_bn'] = dict()

    for l in range(-l_max, l_max+1):
        key = str(l)
        w['bn'][key] = BatchNormalization(momentum=momentum)
        use_bias = (l == 0)
        w['weights'][key] = Dense(units=units, use_bias=use_bias)


    for l in range(1,l_max+1):
        key = str(l)
        w['gate_bn'][key] = BatchNormalization(momentum=momentum)
        w['gate_weights'][key] = Dense(units=units)

    return w
"""

def fourier_to_sph_units_converter(units, l_max):
    u = dict()
    v = dict()
    for l in range(l_max+1):
        u[str(l)] = units * (l_max+1-l)
        v[str(l)] = units
    return u, v

def set_fourier_sph_interwiner(units, l_max):
    w = dict()
    w['0'] = Dense(units=units * (l_max+1), use_bias=True)
    for l in range(1, l_max+1):
        w[str(l)] = Dense(units=units * (l_max+1-l), use_bias=False)
    return w

def fourier_sph_layer(x, w, num_sph):
    y = convert_sph_to_fourier(x)
    print('convert to fourier')
    print(y)
    for l in y:
        if l.isnumeric():
            y[l] = w[l](y[l])
    print('fourier weights')
    print(y)
    y = convert_fourier_to_sph(y)
    return y

def fourier_sph_relu(x, n, bn):
    types = []
    Y = []
    num_channels = []
    for l in x:
        if l.isnumeric():
            types.append(int(l))
            Y.append(inverse_sph_fourier_transform(x[l], n))
            num_channels.append(x[l].shape[-1])
    Y = tf.concat(Y, axis=-1)
    if bn is not None:
        Y = bn(Y)
    Y = tf.nn.relu(Y)
    Y = tf.split(Y, num_or_size_splits=num_channels, axis=-1)
    y = dict()
    for i in range(len(types)):
        l = types[i]
        y[str(l)] = sph_fourier_transform(Y[i], l)
    return y


class TFN(tf.keras.Model):
    def __init__(self, num_classes):
        super(TFN, self).__init__()
        self.num_classes = num_classes
        self.dodecahedron = 'pentakis'
        # self.dodecahedron = 'regular'
        self.d = 3
        self.l_max = [3, 3, 3]
        self.l_max_out = [1, 3, 3, 3]
        self.num_shells = [3, 3, 3]
        self.gaussian_scale = []
        for i in range(len(self.num_shells)):
            self.gaussian_scale.append(0.69314718056 * ((self.num_shells[i]) ** 2))
        self.radius = [0.2, 0.40, 0.8]
        self.bounded = [True, True, True]
        self.num_points = [1024, 256, 64, 16]
        self.patch_size = [32, 32, 32]
        self.spacing = [0, 0, 0]

        self.fourier_units = [16, 32, 64]
        self.sph_units = []
        self.num_sph = []
        for i in range(len(self.fourier_units)):
            l = self.l_max_out[i+1]
            ui = int(math.ceil((2.*l+1.)*self.fourier_units[i]/((l+1.)**2)))
            ui, vi = fourier_to_sph_units_converter(ui, l)
            self.sph_units.append(ui)
            self.num_sph.append(vi)


        self.mlp_units = [[32, 32], [64, 64], [128, 256]]
        self.bn_momentum = 0.75
        self.droupout_rate = 0.5

        self.grouping_layers = []
        self.kernel_layers = []
        self.conv_layers = []
        self.wigner = []

        self.eval = []
        self.coeffs = []
        for i in range(len(self.l_max_out)):
            di = tf_wigner_matrix(l_max=self.l_max_out[i])
            self.wigner.append(di)

        for i in range(len(self.radius)):
            gi = GroupPoints(radius=self.radius[i],
                             patch_size_source=self.patch_size[i],
                             spacing_source=self.spacing[i])
            self.grouping_layers.append(gi)

            ki = SphericalHarmonicsGaussianKernels(l_max=self.l_max[i],
                                                   gaussian_scale=self.gaussian_scale[i],
                                                   num_shells=self.num_shells[i],
                                                   bound=self.bounded[i])

            """
            ci = SphTensorToFourierIrreps(l_list1=[l for l in range(self.l_max_out[i]+1)],
                                          l_list2=[l for l in range(self.l_max[i]+1)],
                                          out_types=[l for l in range(self.l_max_out[i+1]+1)])
            """

            ci = LocallyAlignedConv(l_list1=[l for l in range(self.l_max_out[i]+1)],
                                    l_list2=[l for l in range(self.l_max[i]+1)],
                                    out_types=[l for l in range(self.l_max_out[i+1]+1)])


            self.kernel_layers.append(ki)
            self.conv_layers.append(ci)


            # self.eval.append(DodecahedronEval(l_max=self.l_max_out[i], dodecahedron=self.dodecahedron))
            # self.coeffs.append(DodecahedronCoeffs(l_max=self.l_max_out[i], dodecahedron=self.dodecahedron))

        self.weights_ = []
        self.fourier_to_sph_weights = []
        self.bn = []

        for i in range(len(self.radius)):
            l = self.l_max_out[i+1]
            types = [str(l) for l in range(0, l+1)]
            self.weights_.append(set_fourier_sph_interwiner(units=self.fourier_units[i], l_max=self.l_max_out[i+1]))
            # self.fourier_weights.append(set_intertwiners(self.fourier_units[i], types))
            # self.fourier_to_sph_weights.append(set_intertwiners(self.sph_units[i], types))
            self.bn.append(BatchNormalization(momentum=self.bn_momentum))

        self.fc1_units = 512
        self.fc2_units = 256

        self.fc1 = Dense(units=self.fc1_units, activation=None)
        self.bn_fc1 = BatchNormalization(momentum=self.bn_momentum)
        self.fc2 = Dense(units=self.fc2_units, activation=None)
        self.bn_fc2 = BatchNormalization(momentum=self.bn_momentum)
        self.softmax = Dense(units=self.num_classes, activation='softmax')
        self.S2 = tf_fibonnacci_sphere_sampling(64)

    def call(self, x):
        n0 = x[1]
        x = x[0]
        n0 = center_orientation(x, n0)
        x = kdtree_indexing(x)
        # x = aligned_kdtree_indexing(x)

        points = [x]
        grouped_points = []
        kernels = []

        for i in range(len(self.radius)):
            pi = kd_pooling_1d(points[-1], int(self.num_points[i] / self.num_points[i + 1]))
            # pi = Jitter(self.jitter_scale[i])(pi)
            points.append(pi)



        frames = []


        yzx = []
        for i in range(len(points)):
            yzx_i = tf.stack([points[i][..., 1], points[i][..., 2], points[i][..., 0]], axis=-1)
            yzx.append(tf.expand_dims(yzx_i, axis=-1))

        local_dirs = [n0]
        frames.append(frame_from_vector(n0))

        for i in range(len(self.radius)):
            gi = self.grouping_layers[i]({"source points": points[i], "target points": points[i + 1]})
            pi = gi["patches source"]
            # ni, _, _, _ = local_direction(pi, points[i + 1], weights=None, center_patches=True, center=None)
            # frames.append(frame_from_vector(ni))
            # frames.append(radial_frames(points[i+1]))
            idx0 = gi["patches idx source"][:, :, 0, ...]
            idx0 = tf.expand_dims(idx0, axis=2)
            ni = tf.gather_nd(local_dirs[-1], idx0)
            ni = ni[:, :, 0, ...]
            frames.append(frame_from_vector(ni))
            pi = tf.einsum('bvij,bvpi->bvpj', frames[-1], pi)
            ki = self.kernel_layers[i]({"patches": pi, "patches dist": gi["patches dist source"]})
            grouped_points.append(gi)
            kernels.append(ki)

        wigner = []
        for i in range(len(frames)):
            # fi = radial_frames(points[i])
            # frames.append(fi)
            # wigner.append(self.wigner[i].compute(tf.transpose(fi, (0, 1, 3, 2))))

            wigner.append(self.wigner[i].compute(frames[i]))


        y = {'0': tf.ones((x.shape[0], x.shape[1], 1, 1)), '1': yzx[0]}
        for i in range(len(self.radius)):
            y["source points"] = points[i]
            y["target points"] = points[i + 1]
            y["patches idx"] = grouped_points[i]["patches idx source"] # shape = (batch_size, nv, patch_size, 2)
            y["patches dist source"] = grouped_points[i]["patches dist source"]
            y["kernels"] = kernels[i]# shape (batch_size, nv, patch_size, nl, nr)


            # y['0'], y['1'] ...
            # y['l'] shape = (batch_size, nv_source, 2l+1, nc)
            # align signal with canonical frame


            if '1' in y:
                y['1'] = tf.concat([y['1'], yzx[i]], axis=-1)
            else:
                y['1'] = yzx[i]


            # Y = rotate_real_sph_signal(y, wigner[i], transpose=True)
            # Y, _ = stack_equivariant_features(y)
            # Y = tf.gather_nd(Y, y["patches idx"]) # shape = (batch_size, nv_target, patch_size, nl, nc)
            y = self.conv_layers[i].compute(x=y, patches_idx=y["patches idx"], K=y["kernels"],
                                            D=wigner[i+1], transpose_D=True)
            print('conv')
            print(y)
            y = fourier_sph_layer(y, self.weights_[i], self.num_sph[i])
            print('sph layer')
            print(y)
            y = fourier_sph_relu(y, 16, bn=self.bn[i])

            """

            print('hhh')
            print(y)
            y = convert_sph_to_fourier(y)
            # conv output a fourier signal

            print('ggg')
            print(y)
            y = apply_intertwiners(y, self.fourier_weights[i])



            y = fourier_relu(y, 16, self.bn[i])

            y = apply_intertwiners(y, self.fourier_to_sph_weights[i])

            print('uuu')
            print(y)
            # convert back to sphere signal
            y = convert_fourier_to_sph(y, self.num_sph[i])
            print('rrr')
            print(y)
            """

            y = rotate_real_sph_signal(y, wigner[i+1], transpose=False)






        y = SphericalHarmonicsEval(l_max=self.l_max_out[-1], base=self.S2).compute(y)

        y = tf.reduce_max(y, axis=1, keepdims=False)
        y = SphericalHarmonicsCoeffs(l_max=self.l_max_out[-1], base=self.S2).compute(y)
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





