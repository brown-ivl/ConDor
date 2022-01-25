import tensorflow as tf
from data_providers.provider import Provider
import os
import time
from datetime import datetime
from utils.data_prep_utils import save_h5_data, save_h5_upsampling
from data_providers.classification_datasets import datsets_list
from utils.losses import hausdorff_distance_l1, hausdorff_distance_l2, chamfer_distance_l1, chamfer_distance_l2, orthogonality_loss

from auto_encoder.tfn_auto_encoder_svd import TFN


from pooling import kdtree_indexing, aligned_kdtree_indexing, kdtree_indexing_, aligned_kdtree_indexing_, kd_pooling_1d


# print('test ', tf.test.is_built_with_cuda())
from tensorflow.keras.layers import LeakyReLU
import numpy as np

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

EPOCHS = 101
TIME = time.time()
DATASET = datsets_list[0]
SHUFFLE = True
BATCH_SIZE = 32
NUM_POINTS = 1024
NUM_ITER = 0
SAVE_WEIGHTS = True

WEIGHTS_PATH = 'E:/Users/Adrien/Documents/results/pose_canonicalization_tfn'
# define alignnet here

inputs = tf.keras.layers.Input(batch_shape=(BATCH_SIZE, NUM_POINTS, 3))
autoencoder = tf.keras.models.Model(inputs=inputs, outputs=TFN(1024)(inputs))

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

def l2_loss_(z, x):
    l2_loss = x - z
    l2_loss = tf.reduce_sum(tf.multiply(l2_loss, l2_loss), axis=-1, keepdims=False)
    l2_loss = tf.sqrt(l2_loss + 0.000000001)
    l2_loss = tf.reduce_mean(l2_loss)
    return l2_loss

def losses_(z, x):
    l2_loss = x - z
    l2_loss = tf.reduce_sum(tf.multiply(l2_loss, l2_loss), axis=-1, keepdims=False)
    mean_root_square = l2_loss
    l2_loss = tf.sqrt(l2_loss + 0.000000001)
    l2_loss = tf.reduce_mean(l2_loss)

    mean_root_square = tf.reduce_sum(mean_root_square, axis=1, keepdims=False)
    mean_root_square = tf.sqrt(mean_root_square + 0.000000001) / x.shape[1]
    mean_root_square = tf.reduce_mean(mean_root_square)

    chamfer_loss = chamfer_distance_l2(z, x)
    hausdorff_loss = hausdorff_distance_l2(z, x)
    # loss = chamfer_loss + 0.2*hausdorff_loss
    # loss = chamfer_loss
    loss = l2_loss
    return loss, l2_loss, mean_root_square, chamfer_loss, hausdorff_loss


@tf.function
def test_step(x):
    print("x_shape")
    print(x.shape)

    x = tf_center(x)

    x1 = tf_random_rotate(x)
    x2 = tf_random_rotate(x)

    x1 = kdtree_indexing(x1)
    x2 = kdtree_indexing(x2)


    # yzx = tf_center(tf.stack([x[..., 1], x[..., 2], x[..., 0]], axis=-1))
    z1, inv1, R1 = autoencoder(x1, training=False)
    z2, inv2, R2 = autoencoder(x2, training=False)

    # R = registration(z, inv)
    z12 = tf.einsum('bij,bvj->bvi', R2, inv1)
    z21 = tf.einsum('bij,bvj->bvi', R1, inv2)
    c12 = chamfer_distance_l2(z12, x2)
    c21 = chamfer_distance_l2(z21, x1)
    c = chamfer_distance_l2(z1, z2)
    loss1, l2_loss1, mean_root_square1, chamfer_loss1, hausdorff_loss1 = losses_(z1, x1)
    loss2, l2_loss2, mean_root_square2, chamfer_loss2, hausdorff_loss2 = losses_(z2, x2)
    consistency_loss = c12 + c21 + c
    loss = loss1 + loss2 + consistency_loss
    l2_loss = 0.5*(l2_loss1 + l2_loss2)
    mean_root_square = 0.5 * (mean_root_square1 + mean_root_square2)
    chamfer_loss = 0.5 * (chamfer_loss1 + chamfer_loss2)
    hausdorff_loss = 0.5 * (hausdorff_loss1 + hausdorff_loss2)

    return loss, consistency_loss, l2_loss, mean_root_square, chamfer_loss, hausdorff_loss



@tf.function
def train_step(x):
    print("x_shape")
    print(x.shape)

    x = tf_center(x)

    x1_ = tf_random_rotate(x)
    x2_ = tf_random_rotate(x)

    x1, idx1, idx_inv1 = kdtree_indexing_(x1_)
    x2, idx2, idx_inv2 = kdtree_indexing_(x2_)

    x12 = tf.gather_nd(x1_, idx2)
    x21 = tf.gather_nd(x2_, idx1)
    # yzx = tf_center(tf.stack([x[..., 1], x[..., 2], x[..., 0]], axis=-1))

    with tf.GradientTape() as tape:


        # yzx = tf_center(tf.stack([x[..., 1], x[..., 2], x[..., 0]], axis=-1))
        z1, inv1, R1 = autoencoder(x1, training=True)
        z2, inv2, R2 = autoencoder(x2, training=True)

        # R = registration(z, inv)
        z21 = tf.einsum('bij,bvj->bvi', R2, inv1)
        z12 = tf.einsum('bij,bvj->bvi', R1, inv2)
        # c12 = chamfer_distance_l2(z21, x2)
        # c21 = chamfer_distance_l2(z12, x1)
        c12 = l2_loss_(z12, x12)
        c21 = l2_loss_(z21, x21)
        inv21 = tf.gather_nd(inv2, idx_inv2)
        inv21 = tf.gather_nd(inv21, idx1)
        c = l2_loss_(inv1, inv21)

        loss1, l2_loss1, mean_root_square1, chamfer_loss1, hausdorff_loss1 = losses_(z1, x1)
        loss2, l2_loss2, mean_root_square2, chamfer_loss2, hausdorff_loss2 = losses_(z2, x2)
        consistency_loss = c12 + c21 + c
        loss = loss1 + loss2 + consistency_loss
        l2_loss = 0.5 * (l2_loss1 + l2_loss2)
        mean_root_square = 0.5 * (mean_root_square1 + mean_root_square2)
        chamfer_loss = 0.5 * (chamfer_loss1 + chamfer_loss2)
        hausdorff_loss = 0.5 * (hausdorff_loss1 + hausdorff_loss2)

        grad = tape.gradient(loss, autoencoder.trainable_variables)
        optimizer.apply_gradients(zip(grad, autoencoder.trainable_variables))

    return loss, consistency_loss, l2_loss, mean_root_square, chamfer_loss, hausdorff_loss


def train(trainset, valset, epochs, TEST):
    for epoch in range(epochs):
        trainset.on_epoch_end()
        start = time.time()
        print('epoch: ', epoch)

        loss_ = 0.
        l2_loss_ = 0.
        mrs_loss_ = 0.
        chamfer_loss_l2 = 0.
        hausdorff_loss_l2 = 0.
        consistency_ = 0.

        k = 0
        for x in trainset:
            # y = kdtree_indexing(x, depth=4)
            # y = kd_pooling_1d(y, int(NUM_POINTS / NUM_POINTS_OUT))
            # y = lexicographic_ordering(y)

            if TEST:
                l, c_l, l2, mrs, cl, hl = test_step(x)
            else:
                l, c_l, l2, mrs, cl, hl = train_step(x)

            l2_loss_ += float(l2)
            mrs_loss_ += float(mrs)
            chamfer_loss_l2 += float(cl)
            hausdorff_loss_l2 += float(hl)
            consistency_ += float(c_l)
            loss_ += float(l)

            k += 1

        chamfer_loss_l2 /= k
        hausdorff_loss_l2 /= k
        loss_ /= k
        l2_loss_ /= k
        mrs_loss_ /= k
        consistency_ /= k

        print('loss: ', loss_, ' concistency ', consistency_ / 3.,  ' l2: ', l2_loss_, ' mrs ', mrs_loss_, ' chamfer_l2: ', chamfer_loss_l2, ' hausdorff_l2: ', hausdorff_loss_l2)
        print(' time: ', time.time()-start)


        if epoch == EPOCHS - 1:
            autoencoder.save_weights(os.path.join(WEIGHTS_PATH, 'weights_0.h5'))

        if epoch % 50 == 0 and epoch > 0:
            autoencoder.save_weights(os.path.join(WEIGHTS_PATH, 'weights_epoch_' + str(epoch) + '.h5'))
            # checkpoint.save(file_prefix=checkpoint_prefix)






train(train_provider, val_provider, EPOCHS, False)
train(train_provider, val_provider, 5, True)

"""
if SAVE_WEIGHTS:
    autoencoder.save_weights(os.path.join(WEIGHTS_PATH, 'weights_0.h5'))
"""