
import os
import pacltune
import hashlib
import json
import tensorflow as tf
import tensorflow_addons as tfa

def set_spec_val(key, default, spec_dict):
    if key in spec_dict.keys():
        return spec_dict[key]
    else:
        return default

class Specification:
    def __init__(self, spec_dict={}):
        self.n_epochs = set_spec_val('n_epochs', 10, spec_dict)
        self.app = set_spec_val('app', 'EfficientNetB0', spec_dict)
        self.lr = set_spec_val('lr', 0.001, spec_dict)
        self.augment = set_spec_val('augment', False, spec_dict)
        self.batch_size = set_spec_val('batch_size', 32, spec_dict)
        self.optimizer = set_spec_val('optimizer', 'Adam', spec_dict)
        self.rep = set_spec_val('rep', 0, spec_dict)
        self.val_fold = set_spec_val('val_fold', 0, spec_dict)
        if 'class_dict_name' in spec_dict.keys():
            print('Using class_dict_name provided in spec_dict (=' + str(spec_dict['class_dict_name']))
            class_dict_name = spec_dict['class_dict_name']
        else:
            class_dict_name = 'class-dict'
        self.class_dict = pacltune.utils.read_class_dict_with_warn(class_dict_name)
        if 'data_version' in pacltune.conf.keys():
            self.data_version = pacltune.conf['data_version']
        if self.optimizer == "AdamW":
            if 'weight_decay' in spec_dict.keys():
                weight_decay = spec_dict['weight_decay']
            else:
                weight_decay = 0.00025
                print('Using AdamW but no weight decay set. Using default of ' + str(weight_decay) + ' for weight decay.')
            self.weight_decay = weight_decay
        if self.optimizer == "SGD":
            if 'momentum' in spec_dict.keys():
                momentum = spec_dict['momentum']
            else:
                momentum = 0.9
                print('Using SDG but no momentum set. Using default of ' + str(momentum) + ' for momentum.')
            self.momentum = momentum
        if 'input_size' in spec_dict.keys():
            print('Using input_size provided in spec_dict (=' + str(spec_dict['input_size']) + ') and not default input size from model.')
            self.input_size = spec_dict['input_size']
    def __str__(self):
        return str(self.__dict__)
    def get_id(self):
        dict_no_epochs = {x: self.__dict__[x] for x in self.__dict__ if x not in {'n_epochs'}}
        return hashlib.md5(json.dumps(dict_no_epochs, sort_keys=True).encode('utf-8')).hexdigest()

def model_from_keras_app(spec):
    print('Creating model from keras app...', end='')
    n_classes = len(spec.__dict__['class_dict'])
    if n_classes < 2:
        raise Exception("Need at least two distinct classes. Please check \'class_dict\'.")
    app_fun = getattr(tf.keras.applications, spec.__dict__['app'])
    if 'input_size' in spec.__dict__.keys():
        input_shape = [spec.input_size, spec.input_size, 3]
    else:
        input_shape = None
    with pacltune.tf_strategy.scope():
        model = app_fun(include_top=True,
                        weights=None,
                        pooling='avg',
                        classes=n_classes,
                        input_shape=input_shape)
    print('[DONE]')
    return model

def dummy_model():
    model = tf.keras.models.Sequential([
      tf.keras.layers.Input(shape=(1,1,3)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(2, use_bias=False)
    ])
    return model

def model_from_app(spec):
    if(spec.__dict__['app'] == "__dummy__"):
        model = dummy_model()
    else:
        model = model_from_keras_app(spec)
    return model

def model_from_file(spec):
    print('Loading model from file...', end='')
    model_file = pacltune.utils.get_model_file(spec.get_id())
    with pacltune.tf_strategy.scope():
        model = tf.keras.models.load_model(model_file)
    print('[DONE]')
    return model

def model_from_id(id):
    print('Loading model from id...', end='')
    model_file = pacltune.utils.get_model_file(id)
    with pacltune.tf_strategy.scope():
        model = tf.keras.models.load_model(model_file)
    spec = pacltune.model.Specification(pacltune.utils.get_spec_dict(id))
    if spec.get_id() != id:
        print('WARNING:pacltune:model id given by folder (=' + id + ') is different to id from new spec (=' + spec.get_id() + ').')
    return model, spec

def model_from_id_for_prediction(id):
    print('Loading model from id...', end='')
    spec = pacltune.model.Specification(pacltune.utils.get_spec_dict(id))
    if spec.get_id() != id:
        print('WARNING:pacltune:model id given by folder (=' + id + ') is different to id from new spec (=' + spec.get_id() + ').')
    model_file = pacltune.utils.get_model_file(id)
    with pacltune.tf_strategy.scope():
        model = model_from_app(spec)
        model.load_weights(model_file)
    return model, spec

def get_optimizer(spec):
    if spec.__dict__['optimizer'] == 'AdamW':
        optimizer = tfa.optimizers.AdamW(weight_decay=spec.__dict__['weight_decay'], learning_rate=spec.__dict__['lr'])
        print('Using AdamW optimizer with weight decay ' + str(spec.__dict__['weight_decay']) + ' and lr ' + str(spec.__dict__['lr']) + '.')
    elif spec.__dict__['optimizer'] == 'SGD':
        optimizer = tf.keras.optimizers.SGD(momentum=spec.__dict__['momentum'], learning_rate=spec.__dict__['lr'])
        print('Using SGD optimizer with momentum ' + str(spec.__dict__['momentum']) + ' and lr ' + str(spec.__dict__['lr']) + '.')
    else:
        optimizer_fun = getattr(tf.keras.optimizers, spec.__dict__['optimizer'])
        optimizer = optimizer_fun(learning_rate=spec.__dict__['lr'])
        print('Using ' + spec.__dict__['optimizer'] + ' optimizer with lr ' + str(spec.__dict__['lr']) + '.')
    return optimizer

def compile_model(model, spec):
    optimizer = get_optimizer(spec)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
