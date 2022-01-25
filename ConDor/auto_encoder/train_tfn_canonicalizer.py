import tensorflow as tf
from data_providers.provider import Provider
import os
import time
from datetime import datetime
from utils.data_prep_utils import save_h5_data, save_h5_upsampling
from data_providers.classification_datasets import datsets_list
from utils.losses import hausdorff_distance_l1, hausdorff_distance_l2, chamfer_distance_l1, chamfer_distance_l2, orthogonality_loss

from auto_encoder.tfn_canonicalizer import TFN


from pooling import kdtree_indexing, aligned_kdtree_indexing, kdtree_indexing_, aligned_kdtree_indexing_, kd_pooling_1d


# print('test ', tf.test.is_built_with_cuda())
from tensorflow.keras.layers import LeakyReLU
import numpy as np

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

EPOCHS = 400
TIME = time.time()
DATASET = datsets_list[0]
SHUFFLE = True
BATCH_SIZE = 32
NUM_POINTS = 1024
NUM_ITER = 0
SAVE_WEIGHTS = True


# define alignnet here

inputs_x = tf.keras.layers.Input(batch_shape=(int(BATCH_SIZE / 2), NUM_POINTS, 3))
inputs_y = tf.keras.layers.Input(batch_shape=(int(BATCH_SIZE / 2), NUM_POINTS, 3))
inputs = [inputs_x, inputs_y]
autoencoder = tf.keras.models.Model(inputs=inputs, outputs=TFN(1024)(inputs))

# autoencoder.load_weights('E:/Users/Adrien/Documents/results/order_canonicalization/weights_0.h5')

# weights = 'ckpt-2.data-00000-of-00001'
# alignnet.load_weights('E:/Users/Adrien/Documents/results/unocs/weights_epoch_160.h5')

optimizer = tf.keras.optimizers.Adam(1e-4)

WEIGHTS_PATH = 'E:/Users/Adrien/Documents/results/pose_and_pts_canonicalization_tfn'
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

@tf.function
def train_step(X):

    # x = tf_random_rotate(x)

    # yzx = tf_center(tf.stack([x[..., 1], x[..., 2], x[..., 0]], axis=-1))

    x_ = tf.split(X, axis=0, num_or_size_splits=2)
    x = x_[0]
    y = x_[1]

    with tf.GradientTape() as tape:
        ie_x, ie_y, ied_x, ied_y, it_x, it_y, itd_x, itd_y = autoencoder([x, y], training=True)

        ied = tf.concat([ied_x, ied_y], axis=0)
        itd = tf.concat([itd_x, itd_y], axis=0)

        l2_loss_ied = X - ied
        l2_loss_ied = tf.reduce_sum(tf.multiply(l2_loss_ied, l2_loss_ied), axis=-1, keepdims=False)
        l2_loss_ied = tf.sqrt(l2_loss_ied + 0.000000001)
        l2_loss_ied = tf.reduce_mean(l2_loss_ied)

        l2_loss_itd = X - itd
        l2_loss_itd = tf.reduce_sum(tf.multiply(l2_loss_itd, l2_loss_itd), axis=-1, keepdims=False)
        l2_loss_itd = tf.sqrt(l2_loss_itd + 0.000000001)
        l2_loss_itd = tf.reduce_mean(l2_loss_itd)

        chamfer_loss_it = chamfer_distance_l2(it_x, it_y)

        # hausdorff_loss = hausdorff_distance_l2(z, x)
        # loss = chamfer_loss + 0.2*hausdorff_loss
        # loss = chamfer_loss
        loss = l2_loss_ied + l2_loss_itd + 0.1*chamfer_loss_it

        grad = tape.gradient(loss, autoencoder.trainable_variables)
        optimizer.apply_gradients(zip(grad, autoencoder.trainable_variables))

    return loss, l2_loss_ied, l2_loss_itd, chamfer_loss_it


def train(trainset, valset, epochs):
    for epoch in range(epochs):

        start = time.time()
        print('epoch: ', epoch)


        loss_ = 0.
        l2_loss_ied_ = 0.
        l2_loss_itd_ = 0.
        chamfer_loss_it_ = 0.

        k = 0
        for x in trainset:
            # y = kdtree_indexing(x, depth=4)
            # y = kd_pooling_1d(y, int(NUM_POINTS / NUM_POINTS_OUT))
            # y = lexicographic_ordering(y)


            loss, l2_loss_ied, l2_loss_itd, chamfer_loss_it = train_step(x)

            loss_ += float(loss)
            l2_loss_ied_ += float(l2_loss_ied)
            l2_loss_itd_ += float(l2_loss_itd)
            chamfer_loss_it_ += float(chamfer_loss_it)




            k += 1
        loss_ /= k
        l2_loss_ied_ /= k
        l2_loss_itd_ /= k
        chamfer_loss_it_ /= k

        print('loss: ', loss_, ' l2_loss_ied: ', l2_loss_ied_,
              ' l2_loss_itd: ', l2_loss_itd_, ' chamfer_loss_it: ', chamfer_loss_it_)
        print(' time: ', time.time()-start)

        if epoch == EPOCHS - 1:
            autoencoder.save_weights(os.path.join(WEIGHTS_PATH, 'weights_0.h5'))
        if epoch % 20 == 0 and epoch > 0:
            # autoencoder.save_weights(os.path.join(WEIGHTS_PATH, 'weights_epoch_' + str(epoch) + '.h5'))
            # checkpoint.save(file_prefix=checkpoint_prefix)
            pass


train(train_provider, val_provider, EPOCHS)

"""
if SAVE_WEIGHTS:
    autoencoder.save_weights(os.path.join(WEIGHTS_PATH, 'weights_0.h5'))
"""