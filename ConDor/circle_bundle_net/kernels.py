import tensorflow as tf
from SO3_CNN.spherical_harmonics_ import tf_spherical_harmonics, tf_legendre_polynomials, tf_complex_powers


from tensorflow.keras.layers import Dense, Layer

import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers.ops import core as core_ops
from spherical_harmonics.clebsch_gordan_decomposition import tf_clebsch_gordan_decomposition

# check normalization pointwise normalization might be better (eg divide by the mass of the central shell
# as in Maron et al (PCNN)
def gaussian_shells(r, scale, num_shells=2, radius=None):
    shells_rad = tf.range(num_shells, dtype=tf.float32) / (num_shells - 1)
    shells_rad = tf.reshape(shells_rad, (1, 1, 1, -1))
    shells = tf.subtract(tf.expand_dims(r, axis=-1), shells_rad)
    shells = tf.exp(-scale * tf.multiply(shells, shells))
    shells_sum = tf.reduce_sum(shells, axis=-1, keepdims=True)
    shells = tf.divide(shells, tf.maximum(shells_sum, 0.000001))
    shells = tf.expand_dims(shells, axis=-2)
    if radius is not None:
        shells = tf.where(tf.expand_dims(r, axis=-1) <= radius, shells, 0.)
    mean = tf.reduce_sum(shells, axis=[-3, -1], keepdims=True)
    mean = tf.reduce_mean(mean, axis=1, keepdims=True)
    shells = tf.divide(shells, mean)
    return shells



class ComplexSphericalHarmonicsKernels:
    def __init__(self, scale, radius, l_max=3, num_shells=2, l_list=None):
        self.scale = scale
        self.radius = radius
        if l_list is None:
            self.l_list = range(l_max+1)
        else:
            self.l_list = l_list
        self.l_max = max(self.l_list)
        self.num_shells = num_shells
        self.P = dict()
        self.P = tf_legendre_polynomials(l_max=3, l_list=None)
    def compute(self, x):
        r2 = tf.reduce_sum(tf.multiply(x, x), axis=-1, keepdims=False)
        u = tf.stack([x[..., 0], x[..., 1]], axis=-1)
        u = tf.math.l2_normalize(u, axis=-1)

        r = tf.sqrt(tf.maximum(r2, 0.000001))
        x = tf.divide(x, tf.expand_dims(r, axis=-1))

        x_iy = tf_complex_powers(u[..., 0], u[..., 1], self.l_max)

        z = x[..., -1]
        P = self.P.compute(z, r2)
        Y = dict()

        for l in self.l_list:
            Pl = tf.cast(P[str(l)], dtype=tf.complex64)
            x_iy_l = x_iy[..., self.l_max - l:self.l_max + l + 1]
            Yl = tf.multiply(Pl, x_iy_l)

            for m in range(2*l+1):
                if str(m-l) not in Y:
                    Y[str(m - l)] = []
                Y[str(m - l)].append(Yl[..., m])
        kernel_dim = []
        kernel_type = []
        Y_ = []
        for m in Y:
            kernel_dim.append(len(Y[m]))
            kernel_type.append(m)
            Y_ += Y[m]

        Y = tf.stack(Y_, axis=-1)
        shells = gaussian_shells(r, self.scale, num_shells=2, radius=None)
        shells = tf.cast(shells, dtype=tf.complex64)
        Y = tf.multiply(shells, tf.expand_dims(Y, axis=-1))

        return Y, kernel_type, kernel_dim







"""
class SphericalHarmonicsKernels:
    def __init__(self, l_max, num_shells, l_list=None, l_max_kernels=None):
        self.l_max = l_max
        self.num_shells = num_shells
        self.l_list = l_list
        self.l_max_kernels = l_max_kernels
        if l_max_kernels is None:
            self.l_max_kernels = l_max
        self.lp = tf_legendre_polynomials(l_max=self.l_max_kernels)
    def compute(self, P, R):


        Z = P[..., -1]
        L = self.lp.compute(Z, tf.ones(Z.shape))
        u = tf.stack([P[..., 0], P[..., 1]], axis=-1)
        u = tf.math.l2_normalize(u, axis=-1, epsilon=1e-12)
        tf_complex_powers()
        for l in range(self.l_max_kernels):
"""

def circle_bundle_convolution(X, K, kernel_type, kernel_dim, l_max, l_list=None):

    if l_list is None:
        l_list = range(-l_max, l_max + 1)
    else:
        l_list = l_list

    Y_ = tf.einsum('bvpnr,bvpmc->bvnmrc', K, X)
    Y_shape = list(Y_.shape)
    Y_shape = Y_shape[:-1]
    Y_shape[-1] = -1
    Y_ = tf.reshape(Y_, Y_shape)
    l_max_X = int(X.shape[-2]/2.)
    Y = dict()
    Y_ = tf.split(Y_, num_or_size_splits=kernel_dim, axis=-3)


    for i in range(len(Y_)):
        n = int(kernel_type[i])
        for m in range(2*l_max_X+1):
            k = n + m - l_max_X
            if k in l_list:
                if str(k) not in Y:
                    Y[str(k)] = []
                Y_nm = Y_[i][..., m, :]
                Y_nm = tf.reshape(Y_nm, (Y_nm.shape[0], Y_nm.shape[1], -1))
                Y[str(k)].append(Y_nm)
    for k in Y:
        Y[k] = tf.concat(Y[k], axis=-1)
    return Y



"""
class CircleBundleConvolution:
    def __init__(self):

    def compute(self, X, K, kernel_type, kernel_dim):
        tf.einsum('bvpnr,bvpmc->bvnmr', K, X)
        l_max 
        Y = dict()
        for
"""






class ComplexDense(Layer):

  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(ComplexDense, self).__init__(
        activity_regularizer=activity_regularizer, **kwargs)

    self.units = int(units) if not isinstance(units, int) else units
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.input_spec = InputSpec(min_ndim=2)
    self.supports_masking = True

  def build(self, input_shape):
    dtype = dtypes.as_dtype(self.dtype or K.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('Unable to build `Dense` layer with non-floating point '
                      'dtype %s' % (dtype,))

    input_shape = tensor_shape.TensorShape(input_shape)
    last_dim = tensor_shape.dimension_value(input_shape[-1])
    if last_dim is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
    self.kernel_re = self.add_weight(
        'kernel_re',
        shape=[last_dim, self.units],
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)
    self.kernel_im = self.add_weight(
        'kernel_im',
        shape=[last_dim, self.units],
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)
    self.kernel = (1./np.sqrt(2.))*tf.complex(self.kernel_re, self.kernel_im)
    if self.use_bias:
      self.bias_re = self.add_weight(
          'bias_re',
          shape=[self.units,],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=self.dtype,
          trainable=True)
      self.bias_im = self.add_weight(
          'bias_im',
          shape=[self.units,],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=self.dtype,
          trainable=True)
      self.bias = (1./np.sqrt(2.))*tf.complex(self.bias_re, self.bias_im)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):

    """
    return core_ops.dense(
        inputs,
        self.kernel,
        self.bias,
        self.activation,
        dtype=self._compute_dtype_object)
    """

    y = tf.einsum('...j,ji->...i', inputs, self.kernel)
    if self.bias is not None:
        y = tf.nn.bias_add(y, self.bias)
    return y

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    return input_shape[:-1].concatenate(self.units)

  def get_config(self):
    config = super(Dense, self).get_config()
    config.update({
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint)
    })
    return config

def sphere_to_circle_idx(l_list):
    l_list_ = l_list.copy()
    l_list_.sort()
    stacked_sphere_idx = dict()
    k = 0
    for l in l_list_:
        for m_ in range(2*l+1):
            m = m_ - l
            stacked_sphere_idx[str(l) + ',' + str(m)] = k
            k += 1

    circle_idx_tmp = dict()
    for l in l_list_:
        if 0 not in circle_idx_tmp:
            circle_idx_tmp[0] = []
        circle_idx_tmp[0].append(stacked_sphere_idx[str(l) + ',' + '0'])
        for m in range(1, l+1):
            if m not in circle_idx_tmp:
                circle_idx_tmp[m] = []
            if -m not in circle_idx_tmp:
                circle_idx_tmp[-m] = []
            circle_idx_tmp[m].append(stacked_sphere_idx[str(l) + ',' + str(m)])
            circle_idx_tmp[-m].append(stacked_sphere_idx[str(l) + ',' + str(-m)])

    circle_types = list(circle_idx_tmp.keys())
    circle_types.sort()

    l_max = max(circle_types)
    circle_dims = []
    for i in range(len(circle_types)):
        m = circle_types[i]
        circle_dims.append(len(circle_idx_tmp[m]))
    circle_idx = circle_idx_tmp[0]
    for m in range(1, l_max+1):
        circle_idx = circle_idx + circle_idx_tmp[m]
        circle_idx = circle_idx_tmp[-m] + circle_idx

    circle_idx = np.array(circle_idx, dtype=np.int32)
    circle_idx = tf.convert_to_tensor(circle_idx, dtype=tf.int32)

    return circle_idx, circle_dims, circle_types


class RealSphericalHarmonicsKernels:
    def __init__(self, scale, radius, l_max=3, num_shells=2, l_list=None):
        self.scale = scale
        self.radius = radius
        if l_list is None:
            self.l_list = range(l_max+1)
        else:
            self.l_list = l_list
        self.l_max = max(self.l_list)
        self.l_list.sort()
        self.num_shells = num_shells
        self.SH = tf_spherical_harmonics(l_max=l_max, l_list=l_list)

        circle_idx, circle_dims, circle_types = sphere_to_circle_idx(self.l_list)

        self.idx = circle_idx
        self.kernel_dim = circle_dims
        self.kernel_type = circle_types

    def compute(self, x):
        r2 = tf.reduce_sum(tf.multiply(x, x), axis=-1, keepdims=False)
        r = tf.sqrt(tf.maximum(r2, 0.000001))
        x = tf.divide(x, tf.expand_dims(r, axis=-1))
        y = self.SH.compute(x)
        Y = []
        for l in self.l_list:
            Y.append(y[str(l)])
        Y = tf.concat(Y, axis=-1)
        Y = tf.gather(params=Y, indices=self.idx, axis=-1)
        shells = gaussian_shells(r, self.scale, num_shells=self.num_shells, radius=None)
        Y = tf.multiply(shells, tf.expand_dims(Y, axis=-1))
        return Y

def rotate_complex_sph_signal(x, D, adjoint=False):
    y = dict()
    for l in x:
        if l.isnumeric():
            y[l] = tf.matmul(D[l], x[l], adjoint_a=adjoint)
    return y

def rotate_real_sph_signal(x, D, transpose=False):
    y = dict()
    for l in x:
        if l.isnumeric():
            y[l] = tf.matmul(D[l], x[l], transpose_a=transpose)
    return y

def rotate_stacked_real_sph_signal(x, D, transpose=False):
    y = dict()
    for l in x:
        if l.isnumeric():
            y[l] = tf.matmul(D[l], x[l], transpose_a=transpose)
    return y

def convert_sph_to_fourier(x):
    y = dict()
    for l in x:
        if l.isnumeric():
            l_ = int(l)
            for m in range(0, l_+1):
                y[str(m)] = []
    for l in x:
        if l.isnumeric():
            l_ = int(l)
            if l == '0':
                y['0'].append(tf.expand_dims(x[l][..., 0, :], axis=-2))
            else:
                y['0'].append(tf.expand_dims(x[l][..., l_, :], axis=-2))
                for m in range(1, l_+1):
                    ym = tf.stack([x[l][..., l_-m, :], x[l][..., m+l_, :]], axis=-2)
                    y[str(m)].append(ym)
    for m in y:
        y[m] = tf.concat(y[m], axis=-1)
    return y

def convert_fourier_to_sph(x):
    l_max = 0
    types = []
    for l in x:
        if l.isnumeric():
            types.append(int(l))
            l_max = max(l_max, int(l))
    num_channels = x[str(l_max)].shape[-1]
    for l in types:
        split_size = [num_channels] * int(x[str(l)].shape[-1] / num_channels)
        x[str(l)] = tf.split(x[str(l)], num_or_size_splits=split_size, axis=-1)

    y = dict()
    for l in types:
        yl0 = x['0'].pop()
        y[str(l)] = [yl0[..., 0, :]]
        for m in range(1, l+1):
            yl_mm = x[str(m)].pop()
            yl_m = yl_mm[..., 0, :]
            ylm = yl_mm[..., 1, :]
            y[str(l)] = [yl_m] + y[str(l)] + [ylm]

    for l in types:
        y[str(l)] = tf.stack(y[str(l)], axis=-2)
    return y

def convert_fourier_to_sph_(x, num_sph):




    fourier_types = []
    for l in x:
        if l.isnumeric():
            fourier_types.append(int(l))
    fourier_types.sort()



    fourier_splits = dict()
    sph_type = dict()
    for m in fourier_types:
        fourier_splits[m] = []
        sph_type[m] = []
        for l in num_sph:
            if int(l) >= m:
                fourier_splits[m].append(num_sph[l])
                sph_type[m].append(l)


    fourier_features = []

    for m in fourier_types:
        print(m)
        print(x[str(m)])
        print(fourier_splits[m])
        fourier_features.append(tf.split(x[str(m)], num_or_size_splits=fourier_splits[m], axis=-1))

    print(fourier_types)
    y = dict()
    for l in num_sph:
        for m in fourier_types:
            if m == 0:
                yl0 = fourier_features[0].pop(0)
                y[l] = [yl0]
            elif m <= int(l):
                ylM = fourier_features[m].pop(0)
                ylm = tf.expand_dims(ylM[..., 0, :], axis=-2)
                yl_m = tf.expand_dims(ylM[..., 1, :], axis=-2)
                y[l] = [yl_m] + y[l] + [ylm]
        y[l] = tf.concat(y[l], axis=-2)

    return y


class LocallyAlignedConv:
    def __init__(self, l_list1, l_list2, out_types):
        super(LocallyAlignedConv, self).__init__()

        self.l_list1 = l_list1
        self.l_list2 = l_list2
        self.l_max1 = max(l_list1)
        self.l_max2 = max(l_list2)
        self.l_max = max(self.l_max1, self.l_max2)
        self.l_max_out = self.l_max1 + self.l_max2
        if out_types is not None:
            self.l_max_out = max(out_types)





        self.split_size1 = []
        for l in self.l_list1:
            self.split_size1.append(2 * l + 1)
        self.split_size2 = []
        for l in self.l_list2:
            self.split_size2.append(2 * l + 1)

        self.Q = tf_clebsch_gordan_decomposition(l_max=self.l_max,
                                                 sparse=False,
                                                 output_type='dict',
                                                 l_max_out=self.l_max_out)

    def compute(self, x, patches_idx, K, D, transpose_D=True):
        batch_size = K.shape[0]
        num_points = K.shape[1]
        patch_size = K.shape[2]
        num_shells = K.shape[-1]
        # stack x
        types = []
        num_channels = []
        split_size = []
        X = []
        for l in x:
            if l.isnumeric():
                types.append(int(l))
                num_channels.append(x[l].shape[-1])
                split_size.append((2*int(l)+1)*x[l].shape[-1])
                X.append(tf.reshape(x[l], (batch_size, x[l].shape[1], -1)))
        X = tf.concat(X, axis=-1)
        X = tf.gather_nd(X, patches_idx)


        num_points_target = X.shape[1]

        y = tf.einsum('bvpmr,bvpc->bvcmr', K, X)

        Y = tf.split(y, num_or_size_splits=split_size, axis=2)


        for i in range(len(split_size)):
            yi = tf.reshape(Y[i], (batch_size, num_points, 2*types[i] + 1, num_channels[i], K.shape[-2], K.shape[-1]))
            yi = tf.transpose(yi, (0, 1, 2, 4, 3, 5))
            if D is not None:
                yi = tf.reshape(yi, (batch_size, num_points, 2*types[i] + 1, -1))
                yi = tf.matmul(D[str(types[i])], yi, transpose_a=transpose_D)
            Y[i] = tf.reshape(yi, (batch_size, num_points, 2*types[i] + 1, K.shape[-2], -1))

        y_cg = []
        y = dict()
        for i in range(len(Y)):
            l1 = self.l_list1[i]
            yi = tf.split(Y[i], num_or_size_splits=self.split_size2, axis=3)
            for j in range(len(self.split_size2)):
                l2 = self.l_list2[j]
                yij = yi[j]
                if l2 == 0:
                    if str(l1) not in y:
                        y[str(l1)] = []
                    y[str(l1)].append(yij[:, :, :, 0, :])
                elif l1 == 0:
                    if str(l2) not in y:
                        y[str(l2)] = []
                    y[str(l2)].append(yij[:, :, 0, :, :])
                else:
                    y_cg.append(yij)



        y_cg = self.Q.decompose(y_cg)


        for J in y_cg:
            if J not in y:
                y[J] = []
            y[J].append(y_cg[J])
        for J in y:
            y[J] = tf.concat(y[J], axis=-1)
        return y




class SphTensorToFourierIrreps:
    def __init__(self, l_list1, l_list2, out_types):
        self.l_list1 = l_list1.copy()
        self.l_list1.sort()
        self.l_list2 = l_list2.copy()
        self.l_list2.sort()
        self.out_types = out_types.copy()
        self.out_types.sort()

        self.gather_blocs_idx = []
        self.split_blocs = []
        self.split_types = []
        self.fourier_types_out = []
        self.gather_types_idx = []

        self.o_idx = []
        self.cacb_idx = []
        self.sasb_idx = []
        self.sacb_idx = []
        self.sbca_idx = []

        self.o_types = []
        self.capb_types = []
        self.camb_types = []
        self.sapb_types = []
        self.samb_types = []

        self.sin_sign = []

        self.tensor_idx = dict()
        self.tensor_idx_inv = dict()
        k = 0
        for l1 in self.l_list1:
            for m1 in range(-l1, l1+1):
                for l2 in self.l_list2:
                    for m2 in range(-l2, l2 + 1):
                        if m1 == 0 or m2 == 0:
                            self.o_idx.append(k)
                        if m1 == 0:
                            self.o_types.append(m2)
                        elif m2 == 0:
                            self.o_types.append(m1)
                        self.tensor_idx[(l1, m1, l2, m2)] = k
                        self.tensor_idx_inv[k] = (l1, m1, l2, m2)
                        k += 1
        tensor_size = k
        for l1 in self.l_list1:
            for m1 in range(1, l1+1):
                for l2 in self.l_list2:
                    for m2 in range(1, l2+1):
                        self.cacb_idx.append(self.tensor_idx[(l1, m1, l2, m2)])
                        self.sasb_idx.append(self.tensor_idx[(l1, -m1, l2, -m2)])
                        self.sacb_idx.append(self.tensor_idx[(l1, -m1, l2, m2)])
                        self.sbca_idx.append(self.tensor_idx[(l1, m1, l2, -m2)])
                        self.capb_types.append(m1 + m2)
                        self.camb_types.append(abs(m1 - m2))
                        self.sapb_types.append(-(m1 + m2))
                        self.samb_types.append(-abs(m1 - m2))
                        if m1 - m2 >= 0:
                            self.sin_sign.append(1.)
                        else:
                            self.sin_sign.append(-1.)
        self.split_blocs = [len(self.o_idx), len(self.cacb_idx), len(self.sasb_idx),
                            len(self.sacb_idx), len(self.sbca_idx)]
        self.gather_blocs_idx = self.o_idx + self.cacb_idx + self.sasb_idx + self.sacb_idx + self.sbca_idx
        self.gather_blocs_idx = tf.convert_to_tensor(np.array(self.gather_blocs_idx, dtype=np.int32), tf.int32)
        self.sin_sign = tf.convert_to_tensor(np.array(self.sin_sign, dtype=np.float32), dtype=tf.float32)
        self.sin_sign = tf.reshape(self.sin_sign, (1, 1, -1, 1))

        fourier_types = self.o_types + self.capb_types + self.camb_types + self.sapb_types + self.samb_types
        fourier_types_out = []
        out_idx = []


        for k in range(len(fourier_types)):
            m = fourier_types[k]
            if abs(m) in self.out_types:
                fourier_types_out.append(m)
                out_idx.append(k)

        unique_types_out = list(set(fourier_types_out))
        unique_types_out.sort()
        self.fourier_types_out = unique_types_out
        for i in range(len(unique_types_out)):
            m = unique_types_out[i]
            self.split_types.append(fourier_types_out.count(m))

        fourier_types_out = np.array(fourier_types_out, dtype=np.int32)
        types_argsort = np.argsort(fourier_types_out)
        out_idx = np.array(out_idx, dtype=np.int32)
        out_idx = out_idx[types_argsort]
        self.gather_types_idx = tf.convert_to_tensor(out_idx, dtype=tf.int32)



    def decompose(self, x):
        x_shape = list(x.shape)
        y = tf.reshape(x, [x_shape[0], x_shape[1], -1, x_shape[-1]])
        y = tf.gather(y, self.gather_blocs_idx, axis=-2)
        y = tf.split(y, num_or_size_splits=self.split_blocs, axis=-2)
        o = y[0]
        cacb = y[1]
        sasb = y[2]
        sacb = y[3]
        sbca = y[4]

        capb = cacb - sasb
        camb = cacb + sasb
        sapb = sacb + sbca
        samb = sacb - sbca

        samb = tf.multiply(self.sin_sign, samb)

        y = tf.concat([o, capb, camb, sapb, samb], axis=-2)
        y = tf.gather(y, self.gather_types_idx, axis=-2)
        Y = tf.split(y, num_or_size_splits=self.split_types, axis=-2)
        y_ = dict()
        for i in range(len(Y)):
            l = self.fourier_types_out[i]
            y_[l] = Y[i]
        y = dict()
        types = list(set([abs(l) for l in self.fourier_types_out]))
        for m in types:
            if m == 0:
                y['0'] = tf.reshape(y_[0], [x_shape[0], x_shape[1], 1, -1])
            else:
                ym = tf.stack([y_[-m], y_[m]], axis=-3)
                y[str(m)] = tf.reshape(ym, [x_shape[0], x_shape[1], 2, -1])
        return y

    def convolution(self, X, K, D=None):
        """
        :param X: Input signal patches
        :param K: Sph Kernel
        :param D: Optional Wigner matrix
        :return:
        """
        y = tf.einsum('bvpmr,bvpnc->bvnmrc', K, X)
        if D is not None:
            X_dims = []

            for i in range(len(self.l_list1)):
                X_dims.append(2 * self.l_list1[i] + 1)
            y_shape = list(y.shape)
            y_shape = y_shape[:-1]
            y_shape[-1] = -1
            y = tf.reshape(y, (y_shape[0], y_shape[1], y_shape[2], -1))
            y = tf.split(y, num_or_size_splits=X_dims, axis=-2)
            for i in range(len(y)):
                l = self.l_list1[i]
                y[i] = tf.matmul(D[str(l)], y[i])
            y = tf.concat(y, axis=2)
            y = tf.reshape(y, y_shape)
        y = self.decompose(y)
        # y['0'] shape = (batch_size, nv, 1, nc0)
        # y['m'] shape = (batch_size, nv, 2, ncm)
        return y















def apply_intertwiners(x, layers):
    y = dict()
    for l in x:
        if l.isnumeric():
            y[l] = layers[l](x[l])
    return y


def inverse_sph_fourier_transform(x, n):


    l_max = int(x.shape[-2] / 2.)

    if l_max == 0:
        return tf.tile(x, (1, 1, n, 1))

    t = (2. * np.pi / n) * tf.range(n, dtype=tf.float32)
    k = tf.range(start=1, limit=l_max + 1, dtype=tf.float32)
    t = tf.expand_dims(t, axis=-1)
    k = tf.expand_dims(k, axis=0)
    t = tf.multiply(t, k)
    c = tf.cos(t)
    s = tf.sin(t)
    shape = list(s.shape)
    shape[-1] = 1
    o = tf.ones(shape)
    e = tf.concat([s, o, c], axis=-1)
    y = tf.einsum('ij,...jc->...ic', e, x)
    return y


def sph_fourier_transform(x, l_max):
    n = x.shape[-2]
    if l_max == 0:
        tf.reduce_mean(x, axis=-2, keepdims=True)

    t = (2. * np.pi / n) * tf.range(n, dtype=tf.float32)
    k = tf.range(start=1, limit=l_max + 1, dtype=tf.float32)
    t = tf.expand_dims(t, axis=-1)
    k = tf.expand_dims(k, axis=0)
    t = tf.multiply(t, k)
    c = tf.cos(t)
    s = tf.sin(t)
    shape = list(s.shape)
    shape[-1] = 1
    o = tf.ones(shape)
    e = tf.concat([s, o, c], axis=-1) / n
    y = tf.einsum('ij,...ic->...jc', e, x)
    return y



def inverse_fourier_transform(x, n):
    x_types = []
    for l in x:
        if l.isnumeric():
            x_types.append(int(l))
    l_max = max(x_types)

    if l_max == 0:
        return {'0', tf.tile(x['0'], (1, 1, n, 1))}

    t = (2. * np.pi / n) * tf.range(n, dtype=tf.float32)
    k = tf.range(start=1, limit=l_max + 1, dtype=tf.float32)
    t = tf.expand_dims(t, axis=-1)
    k = tf.expand_dims(k, axis=0)
    t = tf.multiply(t, k)

    c = tf.cos(t)
    s = tf.sin(t)
    e = tf.stack([c, s], axis=-1)

    X = []

    for i in range(len(x_types)):
        m = x_types[i]
        if m > 0:
            X.append(x[str(m)])

    X = tf.stack(X, axis=-3)

    y = tf.einsum('iab,...abc->...ic', e, X)
    tiles = [1] * len(list(x['0'].shape))
    tiles[-2] = n
    y = tf.add(tf.tile(x['0'], tiles), y)
    return y


def fourier_transform(x, l_max):

    y0 = tf.reduce_mean(x, axis=-2, keepdims=True)
    y = dict()
    y['0'] = y0
    if l_max == 0:
        return y

    n = x.shape[-2]
    t = (2. * np.pi / n) * tf.range(n, dtype=tf.float32)
    k = tf.range(start=1, limit=l_max + 1, dtype=tf.float32)
    t = tf.expand_dims(t, axis=-1)
    k = tf.expand_dims(k, axis=0)
    t = tf.multiply(t, k)

    c = tf.cos(t)
    s = tf.sin(t)
    e = 2.*tf.stack([c, s], axis=-1) / n

    y_ = tf.einsum('iab,...ic->...abc', e, x)


    # y = dict()
    for m in range(l_max):
        y[str(m+1)] = y_[..., m, :, :]

    return y

def fourier_relu(x, n, bn=None):

    # x['0'] shape = (nb, nv, , 1, nc)
    # x['m'] shape = (nb, nv, , 2, nc)

    x_types = []
    for l in x:
        if l.isnumeric():
            x_types.append(int(l))
    l_max = max(x_types)
    y = tf.nn.relu(inverse_fourier_transform(x, n))

    if bn is not None:
        y = bn(y)

    y = fourier_transform(y, l_max)
    return y



def apply_local_direction_prior_TFN_layer(x, D_source, patches_idx, D_target,
                                          K, kernel_dims, kernel_types,
                                          fourier_weights, sph_weights, num_sph,
                                          out_types=None,
                                          n_circle_samples=16):

    # align data with the canonical frame
    y = rotate_real_sph_signal(x, D_source, transpose=True)

    Y_types = []
    for l in y:
        if l.isnumeric():
            Y_types.append(int(l))
    Y_types.sort()

    Y = [y[str(Y_types[i])] for i in range(len(Y_types))]
    Y = tf.concat(Y, axis=-1)
    Y = tf.gather_nd(Y, patches_idx)
    y = wigner_convolution(Y, K, Y_types, kernel_dims, kernel_types, D_target, out_types=out_types)
    y = apply_intertwiners(y, fourier_weights)
    y = fourier_relu(y, n_circle_samples)
    y = apply_intertwiners(y, sph_weights)
    y = convert_fourier_to_sph(y, num_sph)
    y = rotate_real_sph_signal(y, D_target)
    return y


