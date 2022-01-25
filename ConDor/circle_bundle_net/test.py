import tensorflow as tf
from circle_bundle_net.kernels import ComplexDense, ComplexSphericalHarmonicsKernels, circle_bundle_convolution
from circle_bundle_net.local_alignment import sphere_transport, normals_reorientation, frame_from_vector
from circle_bundle_net.kernels import fourier_relu, fourier_transform, inverse_fourier_transform, SphTensorToFourierIrreps
import numpy as np


"""
shape = (2, 3)
z = tf.complex(tf.ones(shape), tf.zeros(shape))
print(z)


z = ComplexDense(units=2)(z)
print(z)
"""

"""
X = tf.ones((32, 1024, 32, 3))
F = tf.ones((32, 1024, 32, 7, 10), dtype=tf.complex64)

SHK = SphericalHarmonicsKernels(scale=1., radius=1.)
K, kernel_type, kernel_dim = SHK.compute(X)

circle_bundle_convolution(F, K, kernel_type, kernel_dim, l_max=3, l_list=None)
"""
"""
N1 = tf.ones((1, 1, 3))
N2 = tf.ones((1, 1, 1, 3))

R1 = frame_from_vector(N1)
R2 = frame_from_vector(N2)

print(R1.shape)
print(R2.shape)

T = sphere_transport(R1, R2, 1)
print(T)

N1 = tf.stack([N1, -N1], axis=-2)
N2 = tf.stack([N2, -N2], axis=-2)

N21, N22 = normals_reorientation(N1, N2)

print(N21)
print(N22)

"""


def fourier_(x, n, bn=None):

    # x['0'] shape = (nb, nv, , 1, nc)
    # x['m'] shape = (nb, nv, , 2, nc)

    x_types = []
    for l in x:
        if l.isnumeric():
            x_types.append(int(l))
    l_max = max(x_types)
    # y = tf.nn.relu(inverse_fourier_transform(x, n))
    y = inverse_fourier_transform(x, n)
    y = y*y
    if bn is not None:
        y = bn(y)

    y = fourier_transform(y, l_max)
    return y

def stack(x):
    x_types = []
    for l in x:
        if l.isnumeric():
            x_types.append(int(l))
    x_types.sort()
    Y = []
    for l in x_types:
        Y.append(x[str(l)])
    return tf.concat(Y, axis=-2)

def tensor(x, y):
    x = stack(x)
    y = stack(y)
    x = tf.expand_dims(x, axis=-2)
    y = tf.expand_dims(y, axis=-3)
    return tf.multiply(x, y)

l_max = 1
F1 = dict()
F1['0'] = np.zeros((1, 1, 1, 1))
for l in range(1, l_max+1):
    F1[str(l)] = np.zeros((1, 1, 2*l+1, 1))

F1['1'][0, 0, 0, 0] = 1.

for l in range(0, l_max+1):
    F1[str(l)] = tf.convert_to_tensor(F1[str(l)], dtype=tf.float32)


F2 = dict()
F2['0'] = np.zeros((1, 1, 1, 1))
for l in range(1, l_max+1):
    F2[str(l)] = np.zeros((1, 1, 2*l+1, 1))

F2['1'][0, 0, 2, 0] = 1.

for l in range(0, l_max+1):
    F2[str(l)] = tf.convert_to_tensor(F2[str(l)], dtype=tf.float32)

# print(fourier_(F1, n=64))

l_list = [l for l in range(l_max+1)]
S = SphTensorToFourierIrreps(l_list1=l_list, l_list2=l_list, out_types=[0, 1, 2, 3])

print(tensor(F1, F2))
print(S.decompose(tensor(F1, F2)))

