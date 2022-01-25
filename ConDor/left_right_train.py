from data_providers.simple_provider import SegmentationProvider
import tensorflow as tf
import multiprocessing

tf.config.set_soft_device_placement(True)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# tf.compat.v1.disable_eager_execution()

print(tf.__version__)



from tensorflow.keras import backend as K
import os
from time import time
import datetime
from tensorflow.python.keras.callbacks import TensorBoard
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.save_model import save_matrix, create_dir, save_training_acc, save_model
"""
from segmentation_network import EquivariantSegPointNet, EquivariantSegNet
from segnet_invar import SegNetInvar
from segmentation_net_best import SegNet
from segmentation_net_func import SegNetFunc
"""
from tensorflow.keras.callbacks import Callback
# from segmentation_norm import SegNetNorm
# from TFN.tfn_gated_seg import SegNetNorm
from TFN.tfn_sht_activation_light_seg import SegNet


network = SegNet
method_name = "TFN_sht"
# weights_path = "E:/Users/Adrien/Documents/results/shapenet_seg_augmented/TFN_gated_2021_01_24_10_03_04/TFN_gated_2021_01_24_10_03_04.h5"
weights_path = None

from data_providers.segmentation_datasets import datasets_list
from utils.pointclouds_utils import pc_batch_preprocess

datasets = datasets_list


MODELS_DIR = 'E:/Users/Adrien/Documents/results'
RESULTS_DIR = 'E:/Users/Adrien/Documents/results'
assert(os.path.isdir(MODELS_DIR))
assert(os.path.isdir(RESULTS_DIR))
SAVE_MODELS = False

num_points = 1024
sample_ratio = 0.25
batch_size = 12
num_epochs = 150
SHUFFLE = True
TEST = False
num_test_passes = 10

def load_dataset(dataset):

    train_files_list = dataset['train_files_list']
    val_files_list = dataset['val_files_list']
    test_files_list = dataset['test_files_list']

    train_data_folder = dataset['train_data_folder']
    val_data_folder = dataset['val_data_folder']
    test_data_folder = dataset['test_data_folder']

    train_preprocessing = dataset['train_preprocessing']
    val_preprocessing = dataset['val_preprocessing']
    test_preprocessing = dataset['test_preprocessing']

    num_parts = dataset['num_parts']
    num_classes = dataset['num_classes']
    parts = dataset['parts']

    # cat_to_labels = dataset['cat_to_labels']
    labels_to_cat = dataset['labels_to_cat']

    train_provider = SegmentationProvider(files_list=train_files_list,
                                          data_path=train_data_folder,
                                          n_points=num_points,
                                          batch_size=batch_size,
                                          preprocess=train_preprocessing,
                                          shuffle=SHUFFLE)

    val_provider = SegmentationProvider(files_list=val_files_list,
                                        data_path=val_data_folder,
                                        n_points=num_points,
                                        batch_size=batch_size,
                                        preprocess=val_preprocessing,
                                        shuffle=SHUFFLE)

    test_provider = SegmentationProvider(files_list=test_files_list,
                                         data_path=test_data_folder,
                                         n_points=num_points,
                                         batch_size=batch_size,
                                         preprocess=test_preprocessing,
                                         shuffle=False)

    return train_provider, val_provider, test_provider

def build_model(network, batch_size, num_points, num_parts, num_classes=None, weights_path=None):
    # loss = tf.keras.backend.categorical_crossentropy

    loss = tf.keras.backend.sparse_categorical_crossentropy

    # loss = tf.keras.backend.categorical_crossentropy

    points_input = tf.keras.layers.Input(batch_shape=(batch_size, num_points, 3))
    # samples = tf.keras.layers.Input(batch_shape=(batch_size, int(sample_ratio*num_points), 3))
    if num_classes is not None:
        class_input = tf.keras.layers.Input(batch_shape=(batch_size, num_classes))
        inputs = [points_input, class_input]
    else:
        inputs = [points_input]
    # inputs = {"points": points_input, "class": class_input}

    model = tf.keras.models.Model(
        inputs=inputs, outputs=network(num_parts)(inputs))
    model.compile(optimizer="Adam", loss=loss, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()], run_eagerly=False)
    model.summary()
    if weights_path is not None:
        model.load_weights(weights_path)
    return model



def train(checkpoint_filepath):
    segmenter = build_model(network, batch_size, num_points, 2,
                            num_classes= 1,
                            weights_path=weights_path)
    metrics = MetricsCallback(provider=test_provider, interval=5)
    callbacks = [metrics]
    if SAVE_MODELS:
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_sparse_categorical_accuracy',
            mode='max',
            save_best_only=True)
        callbacks.append(model_checkpoint_callback)

    if not TEST:
        print('num examples')
        print(train_provider.n_samples)
        print(val_provider.n_samples)
        hist = segmenter.fit(x=train_provider, y=None, batch_size=batch_size, epochs=num_epochs, verbose=2, callbacks=callbacks,
                             validation_split=0.0, validation_data=val_provider, shuffle=True, class_weight=None,
                             sample_weight=None, initial_epoch=0,
                             steps_per_epoch=train_provider.n_samples // batch_size,
                             validation_steps=val_provider.n_samples // batch_size,
                             validation_batch_size=batch_size, validation_freq=1,
                             max_queue_size=10, workers=1, use_multiprocessing=False)
    else:
        hist = None
    return segmenter, hist



def IoU__(one_hot_pred, one_hot, num_parts):
    # pred_sum =
    ones = np.ones(shape=(num_parts, ))
    zeros = np.zeros(shape=(num_parts, ))
    true_sum = np.sum(one_hot, axis=0, keepdims=False)
    parts = np.where(true_sum > 0, ones, zeros)
    num_parts = np.sum(parts)
    Inter = np.multiply(one_hot_pred, one_hot)
    Inter = np.sum(Inter, axis=0, keepdims=False)
    Union = np.maximum(one_hot_pred, one_hot)
    Union = np.sum(Union, axis=0, keepdims=False)

    Union = np.where(Union == 0, ones, Union)

    IoU = np.divide(Inter, Union)

    print('parts ', parts)
    print('num_parts ', num_parts)
    print('union ', Union)
    print('inter ', Inter)
    print('iou ', IoU)

    IoU = np.sum(IoU) / num_parts

    return IoU

# Try Zernike kernels
#  Make a callback for restoring best weights or just get them from early stopping callback via self.best_weights
"""
use with 
tf.keras.callbacks.EarlyStopping(monitor='val_mean_class_iou', patience=200, mode='max', restore_best_weights=True)
"""
class MetricsCallback(Callback):
    def __init__(self, provider, interval=4):
        super(Callback, self).__init__()
        self.interval = interval
        self.provider = provider

    def on_train_begin(self, logs={}):
        self.acc = []
        self.per_class_acc = []
        self.mean_class_acc = []
        self.miou = []
        self.per_class_iou = []
        self.mean_class_iou = []
        self.conf_mat = []

    def on_epoch_end(self, epoch, logs={}):
        """
        if epoch % self.interval == 0:
            acc, per_class_acc, mean_class_acc, mIoU, per_class_iou, mean_class_iou, conf_mat = \
                test(self.model, self.provider, dataset=None)

            self.acc.append(acc)
            self.per_class_acc.append(per_class_acc)
            self.mean_class_acc.append(mean_class_acc)
            self.miou.append(mIoU)
            self.per_class_iou.append(per_class_iou)
            self.mean_class_iou.append(mean_class_iou)
            self.conf_mat.append(conf_mat)

            print('per_class_iou: ', per_class_iou)
            print('mean class iou: ', mean_class_iou)
        """


"""
def batch_iou(pred_parts, true_parts, class_label, seg_parts):
    batch_size_ = true_parts.shape[0]
    ious = np.zeros()
    for j in range(batch_size_):

        total_seen_per_cat[cur_class_labels[j]] += 1.
        total_seg_acc_per_cat[cur_class_labels[j]] += per_instance_part_acc[j]

        segp = pred_parts[j, :]
        segl = true_parts[j, :]
        # cat = seg_label_to_cat[segl[0]]

        cat = class_label[j, :]

        part_ious = [0.0 for _ in range(len(seg_parts[cat]))]
        for l in seg_parts[cat]:
            if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                part_ious[l - seg_parts[cat][0]] = 1.0
            else:
                part_ious[l - seg_parts[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                    np.sum((segl == l) | (segp == l)))
        shape_ious[cat].append(np.mean(part_ious))

def test(segmenter, test_provider):

    for X, y in test_provider:
"""

def test(segmenter, test_provider, dataset=None):
    data, part_labels, class_labels = test_provider.get_data()
    batch_size_ = test_provider.get_batch_size()
    # extend data by batch size

    cat_to_labels = test_provider.cat_to_labels
    # labels_to_cat = test_provider.labels_to_cat

    seg_parts = test_provider.seg_parts

    shape_ious = {cat: [] for cat in cat_to_labels.keys()}
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_parts.keys():

        for label in seg_parts[cat]:
            # print('label', label)
            seg_label_to_cat[label] = cat

    data = np.concatenate([data, data[:batch_size_, ...]], axis=0)
    part_labels = np.concatenate([part_labels, part_labels[:batch_size_, ...]], axis=0)
    class_labels = np.concatenate([class_labels, class_labels[:batch_size_, ...]], axis=0)



    num_samples = data.shape[0]
    num_batches = num_samples // batch_size_
    num_parts = test_provider.get_num_parts()
    num_classes = test_provider.get_num_classes()
    num_points = data.shape[1]
    num_points_target = test_provider.get_num_points()
    preprocess = test_provider.get_preprocess()



    pred_part_labels = np.zeros(shape=(num_samples, num_points_target), dtype=np.int32)
    part_labels_ = np.zeros(shape=(num_samples, num_points_target), dtype=np.int32)

    acc = 0.
    mIoU = 0.
    per_class_iou = np.zeros((num_classes, ))
    total_seen_per_cat = np.zeros((num_classes, ), dtype=np.float32)
    total_seg_acc_per_cat = np.zeros((num_classes, ), dtype=np.float32)
    for i in range(num_batches):

        idx = np.random.permutation(np.arange(num_points))[:num_points_target]
        cur_data = data[i*batch_size_:(i+1)*batch_size_, ...]
        cur_data = cur_data[:, idx, ...]
        cur_part_labels = part_labels[i*batch_size_:(i+1)*batch_size_, ...]
        cur_part_labels = cur_part_labels[:, idx, ...]

        # kd tree idx
        for j in range(len(preprocess)):
            cur_data, cur_part_labels = pc_batch_preprocess(cur_data, y=cur_part_labels, proc=preprocess[j])

        cur_one_hot_part_labels = tf.keras.utils.to_categorical(cur_part_labels, num_classes=num_parts)
        cur_part_labels = cur_part_labels[..., 0]

        part_labels_[i*batch_size_:(i+1)*batch_size_, ...] = cur_part_labels
        cur_class_labels = class_labels[i*batch_size_:(i+1)*batch_size_, ...]
        cur_one_hot_class_labels = tf.keras.utils.to_categorical(cur_class_labels, num_classes=num_classes)
        cur_pred_part_labels = segmenter.predict_on_batch(x=[cur_data, cur_one_hot_class_labels])
        cur_pred_part_labels = np.argmax(cur_pred_part_labels, axis=-1)

        cur_one_hot_pred_part_labels = tf.keras.utils.to_categorical(cur_pred_part_labels, num_classes=num_parts)

        pred_part_labels[i*batch_size_:(i+1)*batch_size_, ...] = cur_pred_part_labels

        acc_ = np.equal(cur_part_labels, cur_pred_part_labels)
        acc_ = acc_.astype(np.float)
        per_instance_part_acc = np.mean(acc_, axis=1)
        acc += np.mean(per_instance_part_acc)

        for j in range(batch_size_):

            total_seen_per_cat[cur_class_labels[j]] += 1.
            total_seg_acc_per_cat[cur_class_labels[j]] += per_instance_part_acc[j]


            segp = cur_pred_part_labels[j, :]
            segl = cur_part_labels[j, :]
            cat = seg_label_to_cat[segl[0]]
            part_ious = [0.0 for _ in range(len(seg_parts[cat]))]
            for l in seg_parts[cat]:
                if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                    part_ious[l - seg_parts[cat][0]] = 1.0
                else:
                    part_ious[l - seg_parts[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                        np.sum((segl == l) | (segp == l)))
            shape_ious[cat].append(np.mean(part_ious))

    all_shape_ious = []
    k = 0
    # print('shape_ious ', shape_ious.keys())
    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            all_shape_ious.append(iou)
        shape_ious[cat] = np.mean(shape_ious[cat])
        per_class_iou[k] = shape_ious[cat]
        k += 1

    mIoU = np.mean(all_shape_ious)
    # print('shape_ious ', list(shape_ious.values()))
    mean_class_iou = np.mean(np.array(list(shape_ious.values())))

    acc /= num_batches
    per_class_acc = np.divide(total_seg_acc_per_cat, total_seen_per_cat)
    mean_class_acc = np.mean(per_class_acc)

    """
        for j in range(batch_size_):
            IoU = IoU_(cur_one_hot_pred_part_labels[j, ...], cur_one_hot_part_labels[j, ...], num_parts)
            mIoU += IoU
            per_class_iou[cur_class_labels[j, 0]] += IoU
            total_seen_per_cat[cur_class_labels[j, 0]] += 1.
            total_seg_acc_per_cat[cur_class_labels[j, 0]] += per_instance_part_acc[j]

    mIoU /= (num_batches*batch_size_)
    per_class_iou = np.divide(per_class_iou, total_seen_per_cat)
    acc /= num_batches
    per_class_acc = np.divide(total_seg_acc_per_cat, total_seen_per_cat)
    mean_class_acc = np.mean(per_class_acc)
    mean_class_iou = np.mean(per_class_iou)
    """

    """
    if dataset is not None:
        print(method_name + ' test_acc on ' + dataset['name'] + ' dataset is ', acc)
    """

    part_labels_ = np.reshape(part_labels_, newshape=(-1, ))
    pred_part_labels = np.reshape(pred_part_labels, newshape=(-1, ))

    conf_mat = confusion_matrix(part_labels_, pred_part_labels)
    return acc, per_class_acc, mean_class_acc, mIoU, per_class_iou, mean_class_iou, conf_mat

def save_model_(dir, method, model, timestamp):
    folder = os.path.join(dir, method['name'] + '_' + timestamp)
    save_model(model, folder)

def save_results_(dir, hist, conf_mat, test_acc, train_time, test_time):
    folder = dir
    save_matrix(os.path.join(folder, 'confusion_matrix.txt'), conf_mat)
    save_training_acc(folder, hist)
    save_matrix(os.path.join(folder, 'test_acc.txt'), np.array([test_acc]))
    save_matrix(os.path.join(folder, 'train_time.txt'), np.array([train_time, test_time]))

def save_train_results(dir, hist, train_time):
    folder = dir
    # save_matrix(os.path.join(folder, 'confusion_matrix.txt'), conf_mat)
    save_training_acc(folder, hist)
    # save_matrix(os.path.join(folder, 'test_acc.txt'), np.array([test_acc]))
    save_matrix(os.path.join(folder, 'train_time.txt'), np.array([train_time]))

def save_test_results(dir, acc, per_class_acc, mean_class_acc,
                      mIoU, per_class_iou, mean_class_iou,
                      conf_mat, test_time):

    save_matrix(os.path.join(dir, 'test_acc.txt'), np.array([acc]))
    save_matrix(os.path.join(dir, 'per_class_acc.txt'), per_class_acc)
    save_matrix(os.path.join(dir, 'mean_class_acc.txt'), np.array([mean_class_acc]))

    save_matrix(os.path.join(dir, 'mIoU.txt'), np.array([mIoU]))
    save_matrix(os.path.join(dir, 'per_class_iou.txt'), per_class_iou)
    save_matrix(os.path.join(dir, 'mean_class_iou.txt'), np.array([mean_class_iou]))

    save_matrix(os.path.join(dir, 'confusion_matrix.txt'), conf_mat)
    save_matrix(os.path.join(dir, 'test_time.txt'), np.array([test_time]))

for dataset in datasets:
    results_dir = os.path.join(RESULTS_DIR, dataset['name'])
    models_dir = os.path.join(MODELS_DIR, dataset['name'])
    create_dir(results_dir)
    create_dir(models_dir)
    train_provider, val_provider, test_provider = load_dataset(dataset)


    timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())
    method_results_dir = os.path.join(results_dir, method_name + '_' + timestamp)
    method_model_dir = os.path.join(results_dir, method_name + '_' + timestamp)

    if SAVE_MODELS:
        create_dir(method_results_dir)
    start_time = time()


    segmenter, hist = train(method_results_dir  + '/' + method_name + '_' + timestamp + '.h5')

    train_time = (time()-start_time) / num_epochs

    if not SAVE_MODELS:
        create_dir(method_results_dir)
    # create_dir(method_model_dir)
    if hist is not None:
        save_train_results(method_results_dir, hist, train_time)

    per_class_iou_mean = np.zeros(dataset['num_classes'])
    mean_class_iou_mean = 0.

    for i in range(num_test_passes):
        start_time = time()
        acc, per_class_acc, mean_class_acc, mIoU, per_class_iou, mean_class_iou, conf_mat = \
            test(segmenter, test_provider, dataset=dataset)
        test_time = (time() - start_time)

        print('mean_class_iou ', mean_class_iou)
        print('per_class_iou_mean ', per_class_iou)
        mean_class_iou_mean += mean_class_iou
        per_class_iou_mean += per_class_iou


    mean_class_iou_mean /= num_test_passes
    per_class_iou_mean /= num_test_passes

    print('mean_class_iou_mean ', mean_class_iou_mean)
    print('per_class_iou_mean ', per_class_iou_mean)

    save_test_results(method_results_dir,
                      acc, per_class_acc, mean_class_acc,
                      mIoU, per_class_iou_mean, mean_class_iou_mean, conf_mat, test_time)

        # save_results(method_results_dir, hist, conf_mat, test_acc, train_time, test_time)


    if SAVE_MODELS:
        path = method_model_dir + '/' + method_name + '_' + timestamp + '_final' + '.h5'
        save_model(path, segmenter)


# tensorboard = TensorBoard(log_dir=log_dir+'log/{}'.format(time()), write_graph=False)
# tensorboard = TensorBoard(log_dir=os.path.join(log_dir, 'log'))