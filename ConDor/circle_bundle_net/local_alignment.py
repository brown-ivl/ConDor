import tensorflow as tf
import numpy as np
from SO3_CNN.spherical_harmonics_ import tf_complex_powers


def rot(a, b):
    """
    :param a: A unit vector a
    :param b: A unit vector b
    :return: The rotation around a x b sending a to b (parallel transport from a to b on the sphere)
    """
    v = tf.linalg.cross(a, b)
    c = 1. + tf.reduce_sum(tf.multiply(a, b), axis=-1, keepdims=False)
    mask = c >= 0.01
    c = tf.maximum(c, 0.01)
    eye = tf.reshape(tf.eye(3), [1]*(len(list(v.shape))-1) + [3, 3])
    z = tf.zeros(c.shape)
    v1 = v[..., 0]
    v2 = v[..., 1]
    v3 = v[..., 2]
    V = tf.stack([z, -v3, v2, v3, z, -v1, -v2, v1, z], axis=-1)
    V = tf.reshape(V, list(v.shape) + [3])
    V2 = tf.matmul(V, V)
    R = eye + V + tf.divide(V2, tf.reshape(c, list(c.shape) + [1, 1]))
    return R, mask

def frame_from_vector(n):
    """
    :param n: A unit vector field
    :return: An orthonormal frame field having n as the 3rd vector
    """
    shape = list(n.shape)
    n = tf.reshape(n, (-1, 3))
    n1 = n[..., 0]
    n2 = n[..., 1]
    n3 = n[..., 2]
    z = tf.zeros(n1.shape)
    R = tf.reshape(tf.stack([n2, -n1, z, n3, z, -n1, z, n3, -n2], axis=-1), list(n1.shape) + [3, 3])
    norm2 = tf.reduce_sum(tf.multiply(R, R), axis=-1, keepdims=False)
    idx = tf.cast(tf.argmax(norm2, axis=-1), dtype=tf.int32)
    idx = tf.stack([tf.range(n.shape[0]), idx], axis=-1)
    u = tf.gather_nd(params=R, indices=idx)
    norm2 = tf.gather_nd(params=norm2, indices=idx)
    norm = tf.sqrt(tf.maximum(norm2, 0.000001))
    u = tf.divide(u, tf.expand_dims(norm, axis=-1))
    v = tf.linalg.cross(n, u)
    R = tf.stack([u, v, n], axis=-1)
    R = tf.reshape(R, shape + [3])
    return R

"""
def sphere_transport(u1, u2, l):
    R = rot(u1, u2)
    R1 = frame_from_vector(u1)
    R2 = frame_from_vector(u2)
    v12 = tf.einsum('...ij,...j->...i', R, R1[..., 1])
    c = tf.multiply(v12, R2[..., 1])
    s = tf.multiply(v12, R2[..., 2])
    return tf_complex_powers(c, -s, l)
"""


def sphere_transport(R1, R2, l):
    """
    :param R1: Frame field
    :param R2: Local frame fields over local patches, usually patches of R1
    :param l: bound for the Fourier expansion of the transport
    :return: Computes the action of the transport on Fourier coefficients of fiber functions
    """
    L = [1]*len(list(R2.shape))
    L.pop(-1)
    L[2] = R2.shape[2]
    N1 = tf.tile(tf.expand_dims(R1[..., -1], axis=2), L)
    R, mask = rot(N1, R2[..., -1])
    u12 = tf.einsum('bvpij,bvj->bvpi', R, R1[..., 0])
    c = tf.reduce_sum(tf.multiply(u12, R2[..., 0]), axis=-1, keepdims=False)
    s = tf.reduce_sum(tf.multiply(u12, R2[..., 1]), axis=-1, keepdims=False)
    return tf_complex_powers(c, -s, l)

# Needs to produce a mask when the propagation is ambiguous eg: N2 nearly orthogonal to N1
def normals_reorientation(N1, N2):
    """
    :param N1: bi directional field
    :param N2: Patches of local bi directional fields, typically local patches of N1
    :return: patches of local fields N21 and N22
    N21 is the propagation of the first direction of N1 along local patches
    N22 is the propagation of the second component of N1 along local patches
    """

    N11 = tf.expand_dims(N1, axis=2)
    c = tf.reduce_sum(tf.multiply(N11[..., 0, :], N2[..., 0, :]), axis=-1, keepdims=False)
    shape = list(N2.shape)
    shape.pop(-2)

    N2 = tf.reshape(N2, (-1, 2, 3))
    idx2 = tf.math.greater(c, 0.)

    idx2 = tf.reshape(idx2, (-1,))
    idx1 = tf.math.logical_not(idx2)

    idx1 = tf.cast(idx1, dtype=tf.int32)
    idx2 = tf.cast(idx2, dtype=tf.int32)
    idx0 = tf.range(idx1.shape[0])
    idx1 = tf.stack([idx0, idx1], axis=-1)
    idx2 = tf.stack([idx0, idx2], axis=-1)

    N21 = tf.gather_nd(N2, indices=idx1)
    N22 = tf.gather_nd(N2, indices=idx2)

    N21 = tf.reshape(N21, shape)
    N22 = tf.reshape(N22, shape)

    return N21, N22


"""
def sphere_transport(R11, R12, R21, R22):
    R1121 = rot(R11[..., -1], R21[..., -1])
    R1121 = rot(R11[..., -1], R21[..., -1])

    u1121 = tf.einsum('bvpij,bvj->bvpi', R1121, R11[..., 0])
    v1121 = tf.einsum('bvpij,bvj->bvpi', R1121, R11[..., 1])
    n1121 = tf.linalg.cross(u1121, v1121)
    c1121 = tf.multiply(n1121, )
"""