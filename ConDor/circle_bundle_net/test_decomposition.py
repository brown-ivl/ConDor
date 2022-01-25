import tensorflow as tf
import numpy as np
from circle_bundle_net.kernels import SphTensorToFourierIrreps
from circle_bundle_net.kernels import rotate_real_sph_signal
from SO3_CNN.tf_wigner import tf_wigner_matrix


def zyz_euler_angles(R):
    cb = R[..., 2, 2]
    print(R)
    print('u')
    # if c2 == 1
    cbeq1 = tf.cast(tf.greater_equal(cb, 0.9999), tf.float32)
    a = tf.atan2(y=-R[..., 0, 1], x=R[..., 1, 1])
    c = tf.zeros(a.shape)
    b = tf.zeros(a.shape)
    zyz_cbeq1 = tf.stack([a, b, c], axis=-1)
    zyz_cbeq1 = tf.multiply(tf.expand_dims(cbeq1, axis=-1), zyz_cbeq1)

    # a = tf.stack([R[..., 0, 2], R[..., 1, 2]], axis=-1)
    # alpha = tf.linalg.l2_normalize(alpha, axis=-1)

    return zyz_cbeq1


def rotate_fourier_features(x, theta, l_max):
    c = tf.cos(theta*tf.range(l_max+1, dtype=tf.float32))
    s = tf.sin(theta * tf.range(l_max + 1, dtype=tf.float32))

    print('yyyyy')
    print(c)
    print(s)

    y = dict()
    for k in x:
        if k.isnumeric():
            l = int(k)
            if l == 0:
                y['0'] = x['0']
            else:

                yl1 = c[l]*x[k][..., 1, :] - s[l]*x[k][..., 0, :]
                yl0 = s[l] * x[k][..., 1, :] + c[l] * x[k][..., 0, :]
                y[k] = tf.stack([yl0, yl1], axis=-2)
    return y



def random_sph_features(l_list):
    y = dict()
    for l in l_list:
        y[str(l)] = tf.random.uniform((1, 1, 2*l+1, 1), dtype=tf.float32)
    return y

def stack_sph_features(x):
    x_types = []
    for l in x:
        if l.isnumeric():
            x_types.append(int(l))
    x_types.sort()
    Y = []
    for l in x_types:
        Y.append(x[str(l)])
    return tf.concat(Y, axis=-2)





l_max = 1
l_list = [l for l in range(l_max+1)]
l_list = [1]

l_max_out = 2*l_max
l_list_out = [l for l in range(l_max_out+1)]

D = tf_wigner_matrix(l_max=l_max)

S = SphTensorToFourierIrreps(l_list1=l_list, l_list2=l_list, out_types=l_list)

theta = 3* np.pi / 7
zyz = np.array([theta, 0., 0.], dtype=np.float32)
zyz = tf.convert_to_tensor(zyz, dtype=tf.float32)
c = np.cos(theta)
s = np.sin(theta)
Rz = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1]], dtype=np.float32)
e3 = np.array([0., 0., 1.], dtype=np.float32)

D1 = D.compute(Rz)
D2 = D.compute_euler_zyz(zyz)

print(D1['1'])
print(D2['1'])

a = random_sph_features(l_list)
b = random_sph_features(l_list)

a = {'1': tf.reshape(tf.constant([1., 0., 0.], dtype=tf.float32), (1, 1, 3, 1))}
b = {'1': tf.reshape(tf.constant([0., 1., 0.], dtype=tf.float32), (1, 1, 3, 1))}

Da = rotate_real_sph_signal(a, D1)
Db = rotate_real_sph_signal(b, D1)

Da = stack_sph_features(Da)
Db = stack_sph_features(Db)

a = stack_sph_features(a)
b = stack_sph_features(b)



DaDb = tf.einsum('...ic,...jc->...ijc', Da, Db)
ab = tf.einsum('...ic,...jc->...ijc', a, b)

y1 = S.decompose(DaDb)
y2 = S.decompose(ab)
y2 = rotate_fourier_features(y2, theta, l_max)

print(y1['1'])
print(y2['1'])

# print(zyz_euler_angles(Rz))

