import tensorflow as tf
from data_providers.provider import Provider
import os
import time
from datetime import datetime
from utils.data_prep_utils import save_h5_data, save_h5_upsampling
from data_providers.classification_datasets import datsets_list
from utils.losses import hausdorff_distance_l1, hausdorff_distance_l2, chamfer_distance_l1, chamfer_distance_l2, orthogonality_loss
from utils.losses import sq_distance_mat
from auto_encoder.tfn_capsules import TFN


from pooling import kdtree_indexing, aligned_kdtree_indexing, kdtree_indexing_, aligned_kdtree_indexing_, kd_pooling_1d


# print('test ', tf.test.is_built_with_cuda())
from tensorflow.keras.layers import LeakyReLU
import numpy as np

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

EPOCHS = 301
TIME = time.time()
DATASET = datsets_list[0]
SHUFFLE = True
BATCH_SIZE = 32
NUM_POINTS = 1024
NUM_ITER = 0
SAVE_WEIGHTS = True

WEIGHTS_PATH = 'E:/Users/Adrien/Documents/results/tfn_capsules_partial'
# define alignnet here


inputs = tf.keras.layers.Input(batch_shape=(BATCH_SIZE, NUM_POINTS, 3))
autoencoder = tf.keras.models.Model(inputs=inputs, outputs=TFN(1024)(inputs))
autoencoder.load_weights('E:/Users/Adrien/Documents/results/tfn_capsules/weights_0.h5')

partial_inputs = tf.keras.layers.Input(batch_shape=(BATCH_SIZE, NUM_POINTS // 2, 3))
partial_autoencoder = tf.keras.models.Model(inputs=partial_inputs, outputs=TFN(512)(partial_inputs))

# autoencoder.load_weights(os.path.join(WEIGHTS_PATH, 'weights_0.h5'))

# weights = 'ckpt-2.data-00000-of-00001'
# alignnet.load_weights('E:/Users/Adrien/Documents/results/unocs/weights_epoch_160.h5')

optimizer = tf.keras.optimizers.Adam(1e-4)


"""
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt.h5")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=alignnet)
"""

"""
def tf_grid(res, batch_size):
    rows = tf.range(0., res, delta=1., dtype=tf.float32) - (res - 1.) / 2.
    cols = tf.range(0., res, delta=1., dtype=tf.float32) - (res - 1.) / 2.
    slices = tf.range(0., res, delta=1., dtype=tf.float32) - (res - 1.) / 2.
    k, i, j = tf.meshgrid(slices, cols, rows, indexing='ij')
    g = tf.stack([k, i, j], axis=-1)

    g = tf.cast(g, dtype=tf.float32) / (res / 2.0)
    g = tf.reshape(g, (-1, 3))
    g = tf.expand_dims(g, axis=0)
    g = tf.tile(g, (batch_size, 1, 1))

    return g
"""

def lexicographic_ordering(x):
    m = tf.reduce_min(x, axis=1, keepdims=True)
    y = tf.subtract(x, m)
    M = tf.reduce_max(y, axis=1, keepdims=True)
    y = tf.multiply(M[..., 2]*M[..., 1], y[..., 0]) + tf.multiply(M[..., 2], y[..., 1]) + y[..., 2]
    batch_idx = tf.range(y.shape[0])
    batch_idx = tf.expand_dims(batch_idx, axis=-1)
    batch_idx = tf.tile(batch_idx, multiples=(1, y.shape[1]))
    idx = tf.argsort(y, axis=1)
    idx = tf.stack([batch_idx, idx], axis=-1)
    return tf.gather_nd(x, idx)


def generate_3d():
    """Generate a 3D random rotation matrix.
    Returns:
        np.matrix: A 3D rotation matrix.
    """
    x1, x2, x3 = np.random.rand(3)
    R = np.matrix([[np.cos(2 * np.pi * x1), np.sin(2 * np.pi * x1), 0],
                   [-np.sin(2 * np.pi * x1), np.cos(2 * np.pi * x1), 0],
                   [0, 0, 1]])
    v = np.matrix([[np.cos(2 * np.pi * x2) * np.sqrt(x3)],
                   [np.sin(2 * np.pi * x2) * np.sqrt(x3)],
                   [np.sqrt(1 - x3)]])
    H = np.eye(3) - 2 * v * v.T
    M = -H * R
    return M

def rotate_point_cloud(arr):
    return np.einsum('ij,vj->vi', generate_3d(), arr)


def tf_random_rotation(shape):
    if isinstance(shape, int):
        shape = [shape]

    batch_size = shape[0]
    t = tf.random.uniform(shape + [3], minval=0., maxval=1.)
    c1 = tf.cos(2 * np.pi * t[:, 0])
    s1 = tf.sin(2 * np.pi * t[:, 0])

    c2 = tf.cos(2 * np.pi * t[:, 1])
    s2 = tf.sin(2 * np.pi * t[:, 1])

    z = tf.zeros(shape)
    o = tf.ones(shape)

    R = tf.stack([c1, s1, z, -s1, c1, z, z, z, o], axis=-1)
    R = tf.reshape(R, shape + [3, 3])

    v1 = tf.sqrt(t[:, -1])
    v3 = tf.sqrt(1-t[:, -1])
    v = tf.stack([c2 * v1, s2 * v1, v3], axis=-1)
    H = tf.tile(tf.expand_dims(tf.eye(3), axis=0), (batch_size, 1, 1)) - 2.* tf.einsum('bi,bj->bij', v, v)
    M = -tf.einsum('bij,bjk->bik', H, R)
    return M

def tf_random_rotate(x):
    R = tf_random_rotation(x.shape[0])
    return tf.einsum('bij,bpj->bpi', R, x)







def load_dataset(dataset):
    batch_size = BATCH_SIZE
    num_points = NUM_POINTS

    train_files_list = dataset['train_files_list']
    val_files_list = dataset['val_files_list']
    test_files_list = dataset['test_files_list']

    train_data_folder = dataset['train_data_folder']
    val_data_folder = dataset['val_data_folder']
    test_data_folder = dataset['test_data_folder']

    train_preprocessing = dataset['train_preprocessing']
    val_preprocessing = dataset['val_preprocessing']
    test_preprocessing = dataset['test_preprocessing']
    NUM_SAMPLES_DOWNSAMPLED = None

    train_provider = Provider(files_list=train_files_list,
                              data_path=train_data_folder,
                              n_points=num_points,
                              n_samples=NUM_SAMPLES_DOWNSAMPLED,
                              batch_size=batch_size,
                              preprocess=train_preprocessing,
                              shuffle=SHUFFLE)

    val_provider = Provider(files_list=val_files_list,
                            data_path=val_data_folder,
                            n_points=num_points,
                            n_samples=NUM_SAMPLES_DOWNSAMPLED,
                            batch_size=batch_size,
                            preprocess=val_preprocessing,
                            shuffle=SHUFFLE)

    test_provider = Provider(files_list=test_files_list,
                             data_path=test_data_folder,
                             n_points=num_points,
                             n_samples=NUM_SAMPLES_DOWNSAMPLED,
                             batch_size=batch_size,
                             preprocess=test_preprocessing,
                             shuffle=False)

    return train_provider, val_provider, test_provider

train_provider, val_provider, test_provider = load_dataset(DATASET)

TEST_BATCH_Y = test_provider.__getitem__(0)
# test_data = test_provider.get_data()
# TEST_BATCH = test_data[:BATCH_SIZE, ...]



# h5_filename = os.path.join(PREDS_PATH, 'test_batch_' + str(TIME) + '.h5')
# save_h5_data(h5_filename, TEST_BATCH)

def tf_center(x):
    c = tf.reduce_mean(x, axis=1, keepdims=True)
    return tf.subtract(x, c)

def tf_dist(x, y):
    d = tf.subtract(x, y)
    d = tf.multiply(d, d)
    d = tf.reduce_sum(d, axis=-1, keepdims=False)
    d = tf.sqrt(d + 0.00001)
    return tf.reduce_mean(d)

def var_normalize(x):
    c = tf.reduce_mean(x, axis=1, keepdims=True)
    x = tf.subtract(x, c)
    n2 = tf.multiply(x, x)
    n2 = tf.reduce_sum(n2, axis=-1, keepdims=True)
    n2 = tf.reduce_mean(n2, axis=1, keepdims=True)
    sigma = tf.sqrt(n2 + 0.000000001)
    x = tf.divide(x, sigma)
    return x

def registration(x, y):
    x = var_normalize(x)
    y = var_normalize(y)
    xyt = tf.einsum('bvi,bvj->bij', x, y)
    s, u, v = tf.linalg.svd(xyt)
    R = tf.matmul(v, u, transpose_b=True)
    return R



def compute_centroids(points, capsules):
    return tf.einsum('bij,bic->bcj', points, capsules)

def localization_loss(points, capsules, centroids=None):
    if centroids is None:
        centroids = compute_centroids(points, capsules)
    D2 = sq_distance_mat(points, centroids)
    l = tf.einsum('bic,bic->bc', capsules, D2)
    return tf.reduce_mean(l)



def equilibrium_loss(unnormalized_capsules):
    a = tf.reduce_mean(unnormalized_capsules, axis=1, keepdims=False)
    am = tf.reduce_mean(a, axis=-1, keepdims=True)
    l = tf.subtract(a, am)
    l = l*l
    return tf.reduce_mean(l)


def repulsive_loss_2(points, capsules, sq_distance_mat_=None):
    if sq_distance_mat_ is None:
        D2 = sq_distance_mat(points, points)
    else:
        D2 = sq_distance_mat_
    l = tf.einsum('bij,bjc->bic', D2, capsules)
    l = tf.einsum('bic,bid->bcd', l, capsules)
    return tf.reduce_mean(l)




@tf.function
def test_step(x):
    print("x_shape")
    print(x.shape)
    x = tf_center(x)
    x = tf_random_rotate(x)
    x = kdtree_indexing(x)
    # yzx = tf_center(tf.stack([x[..., 1], x[..., 2], x[..., 0]], axis=-1))
    caps, inv, basis = autoencoder(x, training=False)
    eps = 1e-6
    caps_sum = tf.reduce_sum(caps, axis=1, keepdims=True)
    normalized_caps = tf.divide(caps, caps_sum + eps)

    centroids = compute_centroids(x, normalized_caps)

    orth_loss = orthogonality_loss(basis)
    s, u, v = tf.linalg.svd(basis)
    orth_basis = tf.matmul(u, v, transpose_b=True)

    y = tf.einsum('bvj,bmj->bvm', inv, orth_basis)
    y = tf.stack([y[..., 2], y[..., 0], y[..., 1]], axis=-1)

    eq_loss = equilibrium_loss(caps)
    loc_loss = localization_loss(x, normalized_caps, centroids)
    caps_chamf_loss = chamfer_distance_l2(x, centroids)

    # R = registration(z, inv)

    l2_loss = x - y
    l2_loss = tf.reduce_sum(tf.multiply(l2_loss, l2_loss), axis=-1, keepdims=False)
    mean_root_square = l2_loss
    l2_loss = tf.sqrt(l2_loss + 0.000000001)
    l2_loss = tf.reduce_mean(l2_loss)

    mean_root_square = tf.reduce_sum(mean_root_square, axis=1, keepdims=False)
    mean_root_square = tf.sqrt(mean_root_square + 0.000000001) / x.shape[1]
    mean_root_square = tf.reduce_mean(mean_root_square)

    chamfer_loss = chamfer_distance_l2(y, x)
    hausdorff_loss = hausdorff_distance_l2(y, x)
    # loss = chamfer_loss + 0.2*hausdorff_loss
    # loss = chamfer_loss
    loss = l2_loss
    return loss, l2_loss, mean_root_square, chamfer_loss, hausdorff_loss, eq_loss, loc_loss, caps_chamf_loss, orth_loss, y


@tf.function
def train_step(x):
    print("x_shape")
    print(x.shape)
    x = tf_center(x)
    x = tf_random_rotate(x)
    x = kdtree_indexing(x)
    # yzx = tf_center(tf.stack([x[..., 1], x[..., 2], x[..., 0]], axis=-1))

    with tf.GradientTape() as tape:
        caps, inv, basis = autoencoder(x, training=True)
        eps = 1e-6
        caps_sum = tf.reduce_sum(caps, axis=1, keepdims=True)
        normalized_caps = tf.divide(caps, caps_sum + eps)

        centroids = compute_centroids(x, normalized_caps)

        # orth_loss = orthogonality_loss(basis)
        s, u, v = tf.linalg.svd(basis)
        orth_basis = tf.matmul(u, v, transpose_b=True)

        orth_loss = tf.reduce_mean(tf.abs(basis - tf.stop_gradient(orth_basis)))

        y = tf.einsum('bvj,bmj->bvm', inv, orth_basis)
        y = tf.stack([y[..., 2], y[..., 0], y[..., 1]], axis=-1)

        eq_loss  = 2.*equilibrium_loss(caps)
        loc_loss = localization_loss(x, normalized_caps, centroids)
        caps_chamf_loss = chamfer_distance_l2(x, centroids)



        l2_loss = x - y
        l2_loss = tf.reduce_sum(tf.multiply(l2_loss, l2_loss), axis=-1, keepdims=False)
        mean_root_square = l2_loss
        l2_loss = tf.sqrt(l2_loss + 0.000000001)
        l2_loss = tf.reduce_mean(l2_loss)

        mean_root_square = tf.reduce_sum(mean_root_square, axis=1, keepdims=False)
        mean_root_square = tf.sqrt(mean_root_square + 0.000000001) / x.shape[1]
        mean_root_square = tf.reduce_mean(mean_root_square)

        chamfer_loss = chamfer_distance_l2(y, x)
        hausdorff_loss = hausdorff_distance_l2(y, x)
        # loss = chamfer_loss + 0.2*hausdorff_loss
        # loss = chamfer_loss
        loss = l2_loss + eq_loss + loc_loss + caps_chamf_loss + orth_loss
        # loss = orth_loss
        grad = tape.gradient(loss, autoencoder.trainable_variables)
        optimizer.apply_gradients(zip(grad, autoencoder.trainable_variables))

    return loss, l2_loss, mean_root_square, chamfer_loss, hausdorff_loss, eq_loss, loc_loss, caps_chamf_loss, orth_loss, y

def tf_random_dir(shape):
    if isinstance(shape, int):
        shape = [shape]
    t = tf.random.uniform(shape + [3], minval=-1., maxval=1.)
    return tf.linalg.l2_normalize(t, axis=-1)

def partial_shapes(x):
    num_points = x.shape[1]
    num_points_partial = num_points // 2
    batch_size = x.shape[0]
    v = tf_random_dir(batch_size)
    vx = tf.einsum('bj,bpj->bp', v, x)
    x_partial_idx = tf.argsort(vx, axis=-1)
    x_partial_idx = x_partial_idx[:, :num_points_partial]
    batch_idx = tf.expand_dims(tf.range(batch_size), axis=1)
    batch_idx = tf.tile(batch_idx, (1, num_points_partial))
    x_partial_idx = tf.stack([batch_idx, x_partial_idx], axis=-1)
    x_partial = tf.gather_nd(x, x_partial_idx)
    x_partial = tf_center(x_partial)
    x_partial, kd_idx, kd_idx_inv = kdtree_indexing_(x_partial)
    return x_partial, x_partial_idx, kd_idx_inv


def l2_loss_(x, y):
    z = x - y
    z = z*z
    z = tf.reduce_sum(z, axis=-1)
    z = tf.sqrt(z + 0.0000001)
    return tf.reduce_mean(z)


@tf.function
def train_step_partial(x, training=True):
    print("x_shape")
    print(x.shape)
    x = tf_center(x)
    x = tf_random_rotate(x)
    x = kdtree_indexing(x)
    # yzx = tf_center(tf.stack([x[..., 1], x[..., 2], x[..., 0]], axis=-1))

    x_partial, x_partial_idx, kd_idx_inv = partial_shapes(x)

    # x_partial = tf_center(x_partial)
    x_restriction = tf.gather_nd(x, x_partial_idx)
    x_restriction = tf_center(x_restriction)

    with tf.GradientTape() as tape:
        caps, inv, basis = autoencoder(x, training=False)
        caps = tf.stop_gradient(caps)
        inv = tf.stop_gradient(inv)
        basis = tf.stop_gradient(basis)
        p_caps, p_inv, p_basis = partial_autoencoder(x_partial, training=training)
        p_caps = tf.gather_nd(p_caps, kd_idx_inv)
        p_inv = tf_center(tf.gather_nd(p_inv, kd_idx_inv))

        caps_restriction = tf.gather_nd(caps, x_partial_idx)
        inv_restriction = tf_center(tf.gather_nd(inv, x_partial_idx))


        inv_l2_loss = l2_loss_(p_inv, tf.stop_gradient(inv_restriction))
        caps_loss = tf.keras.losses.CategoricalCrossentropy()(tf.stop_gradient(caps_restriction), p_caps)

        s, u, v = tf.linalg.svd(p_basis)
        p_orth_basis = tf.matmul(u, v, transpose_b=True)
        orth_loss = tf.reduce_mean(tf.abs(p_basis - tf.stop_gradient(p_orth_basis)))
        p_y = tf.einsum('bvj,bmj->bvm', p_inv, p_orth_basis)
        p_y = tf.stack([p_y[..., 2], p_y[..., 0], p_y[..., 1]], axis=-1)
        l2_loss = l2_loss_(p_y, x_restriction)

        loss = caps_loss + l2_loss + orth_loss + inv_l2_loss
        # loss = orth_loss
        if training:
            grad = tape.gradient(loss, partial_autoencoder.trainable_variables)
            optimizer.apply_gradients(zip(grad, partial_autoencoder.trainable_variables))

    return loss, l2_loss, inv_l2_loss, caps_loss, orth_loss, p_y

def train(trainset, valset, epochs, TEST):
    for epoch in range(epochs):
        trainset.on_epoch_end()
        start = time.time()
        print('epoch: ', epoch)

        l2_loss_ = 0.
        inv_l2_ = 0.
        caps_loss = 0.
        orth_loss = 0.
        loss_ = 0.

        k = 0
        for x in trainset:
            # y = kdtree_indexing(x, depth=4)
            # y = kd_pooling_1d(y, int(NUM_POINTS / NUM_POINTS_OUT))
            # y = lexicographic_ordering(y)

            if TEST:
                l, l2, inv_l2, cl, o, z = train_step_partial(x, training=False)
            else:
                l, l2, inv_l2, cl, o, z = train_step_partial(x)

            l2_loss_ += float(l2)
            inv_l2_ += float(inv_l2)
            caps_loss += float(cl)
            orth_loss += float(o)
            loss_ += float(l)



            k += 1

        l2_loss_ /= k
        inv_l2_ /= k
        caps_loss /= k
        orth_loss /= k
        loss_ /= k

        print('loss: ', loss_, ' l2_loss: ', l2_loss_, 'inv_l2: ', inv_l2_, 'caps loss: ', caps_loss, 'orth: ', orth_loss)
        print(' time: ', time.time()-start)


        if epoch == EPOCHS - 1:
            partial_autoencoder.save_weights(os.path.join(WEIGHTS_PATH, 'weights_0.h5'))

        if epoch % 50 == 0 and epoch > 0:
            partial_autoencoder.save_weights(os.path.join(WEIGHTS_PATH, 'weights_epoch_' + str(epoch) + '.h5'))
            # checkpoint.save(file_prefix=checkpoint_prefix)




train(train_provider, val_provider, EPOCHS, False)
train(train_provider, val_provider, 5, True)

"""
if SAVE_WEIGHTS:
    autoencoder.save_weights(os.path.join(WEIGHTS_PATH, 'weights_0.h5'))
"""