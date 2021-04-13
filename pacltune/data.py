
import pacltune
import tensorflow as tf
import pandas as pd
import os

def get_label(cal_row, class_dict):
    one_hot = cal_row[0] == list(class_dict.keys())
    return one_hot

def preprocess_img(img, input_size):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [input_size, input_size])
    img = (img / 255.0)
    return img

def preprocess_row_for_img(tab_row, input_size, col_idx):
    img = tf.io.read_file(pacltune.conf['path_to_patches'] + '/' + tab_row[col_idx])
    img = preprocess_img(img, input_size)
    return img

def preprocess_row(cal_row, class_dict, input_size, augment_fun):
    label = get_label(cal_row, class_dict)
    img = preprocess_row_for_img(cal_row, input_size, col_idx=1)
    if augment_fun is not None:
         img = augment_fun(img)
    return img, label

def get_shuffle_buffer_size(batch_size):
    if 'shuffle_buffer_size' in pacltune.conf.keys():
        shuffle_buffer_size = pacltune.conf['shuffle_buffer_size']
    else:
        shuffle_buffer_size = batch_size * 8
    print('Shuffling data with buffer_size = ' + str(shuffle_buffer_size), '.')
    return shuffle_buffer_size

def get_data_cache_file(cachename_add, input_size):
    if 'data_cache_path' in pacltune.conf.keys():
        data_cache_path = pacltune.conf['data_cache_path']
    else:
        data_cache_path = 'output/cache'
    if data_cache_path == '':
        print("Chaching data to RAM.")
        return ''
    if not os.path.isdir(data_cache_path):
        os.makedirs(data_cache_path)
    data_cache_file = data_cache_path + '/cache_tfdata_' + cachename_add + str(input_size)
    print("Chaching data to \'" + data_cache_file + '\'.')
    return data_cache_file

def create_tfdata(data_rows, training, class_dict, input_size, batch_size, augment):
    tfdata = tf.data.Dataset.from_tensor_slices(data_rows)
    augment_fun = pacltune.augment.get_fun(augment)
    if training:
        shuffle_buffer_size = get_shuffle_buffer_size(batch_size=batch_size)
        tfdata = tfdata.shuffle(buffer_size=shuffle_buffer_size)
        if augment_fun is not None:
            print('Data is augmented.')
    else:
        if augment_fun is not None:
            print('WARNING:pacltune:Created tfdata is not for training but augmentation is used.')
    tfdata = tfdata.map(lambda x : preprocess_row(x, class_dict, input_size, augment_fun), num_parallel_calls=pacltune.AUTOTUNE)
    tfdata = tfdata.batch(batch_size)
    if 'prefetch_size' in pacltune.conf.keys():
        print("Using prefetch size of \'" + str(pacltune.conf['prefetch_size']) + '\'.')
        tfdata = tfdata.prefetch(buffer_size=pacltune.conf['prefetch_size'])
    else:
        tfdata = tfdata.prefetch(buffer_size=pacltune.AUTOTUNE)
    return tfdata

def init(augment=None, input_size=None, class_dict=None, val_fold=None, batch_size=None):
    if not 'path_to_patches' in pacltune.conf.keys():
        raise Exception('Need \'path_to_patches\' in \'project.yml\' to create tfdata.')
    args = locals()
    for k in args.keys():
        if args[k] is None:
            args[k] = pacltune.defaults.__dict__[k]
            print('No value for \'' + k + '\' for function \'init\' provided. Using repective value from default specification (=' + str(args[k]) + ').')
    callibration_patches = pd.read_csv('input/callibration-patches.csv')
    is_training_row = callibration_patches['fold'] != args['val_fold']
    is_val_row = callibration_patches['fold'] == args['val_fold']
    tfdata_train = create_tfdata(data_rows=callibration_patches[is_training_row][['pacl_class', 'file']],
                                 training=True,
                                 class_dict=class_dict,
                                 input_size=args['input_size'],
                                 batch_size=args['batch_size'],
                                 augment=args['augment'])
    tfdata_val = create_tfdata(data_rows=callibration_patches[is_val_row][['pacl_class', 'file']],
                               training=False,
                               class_dict=class_dict,
                               input_size=args['input_size'],
                               batch_size=args['batch_size'],
                               augment=False)
    return tfdata_train, tfdata_val

def init_eval(set, input_size, val_fold, batch_size):
    if set == 'train':
        callibration_patches = pd.read_csv('input/callibration-patches.csv')
        is_train_row = callibration_patches['fold'] != val_fold
        patches_tab = callibration_patches[is_train_row]
    if set == 'val':
        callibration_patches = pd.read_csv('input/callibration-patches.csv')
        is_val_row = callibration_patches['fold'] == val_fold
        patches_tab = callibration_patches[is_val_row]
    if set == 'test':
        patches_tab = pd.read_csv('output/data/test-patches.csv')
    if set != 'train' and set != 'val' and set != 'test':
        raise Exception('Provided set (=' + set +') not supported.')
    tfdata = tf.data.Dataset.from_tensor_slices(patches_tab[['file']])
    tfdata = tfdata.map(lambda x : preprocess_row_for_img(x, input_size, col_idx=0), num_parallel_calls=pacltune.AUTOTUNE)
    tfdata = tfdata.batch(batch_size) # Put before the mapping?
    tfdata = tfdata.prefetch(buffer_size=pacltune.AUTOTUNE)
    return tfdata, patches_tab
