
import pacltune
import os
import sys
import json
import importlib
import pandas as pd
import tensorflow as tf

def create_callbacks(model_id):
    model_logger = tf.keras.callbacks.ModelCheckpoint(filepath=pacltune.utils.get_model_file(model_id),
                                                      verbose=0)
    csv_logger = tf.keras.callbacks.CSVLogger(pacltune.utils.get_history_file(model_id),
                                              append=True)
    return [model_logger, csv_logger]

def fitpacl(spec):
    model_path = pacltune.utils.get_model_path(spec.get_id())
    model_exist = os.path.isdir(model_path)
    if model_exist:
        spec_dict_prev = json.load(open(pacltune.utils.get_spec_file(spec.get_id()), 'r'))
        history_prev = pd.read_csv(pacltune.utils.get_history_file(spec.get_id()))
        n_epochs_done = max(history_prev['epoch']) + 1 # starts with 0
        n_epochs_mis = spec.__dict__['n_epochs'] - min(spec_dict_prev['n_epochs'], n_epochs_done)
        print('Model exists with ' + str(n_epochs_mis) + ' epochs missing.', end = '')
        if n_epochs_done != spec_dict_prev['n_epochs']:
            print(' ' + str(spec_dict_prev['n_epochs'] - n_epochs_done) + '\\' + str(n_epochs_mis) + ' from previous unfinished tuning.')
        else:
            print('')
        if n_epochs_mis < 1:
            return
        json.dump(spec.__dict__, open(pacltune.utils.get_spec_file(spec.get_id()), 'w'))
        initial_epoch = min(spec_dict_prev['n_epochs'], n_epochs_done)
        model = pacltune.model.model_from_file(spec)
    else:
        os.makedirs(model_path)
        with open(pacltune.utils.get_spec_file(spec.get_id()), 'w') as spec_file:
            json.dump(spec.__dict__, spec_file)
        initial_epoch = 0
        model = pacltune.model.model_from_app(spec)
        model = pacltune.model.compile_model(model, spec)
    input_shape = model.get_layer(index=0).input.get_shape().as_list()
    if input_shape[1] != input_shape[2]:
        raise Exception('Model input shape is not quadratic: ' + str(input_shape))
    tfdata_train, tfdata_val = pacltune.data.init(augment=spec.__dict__['augment'],
                                                  input_size=input_shape[1],
                                                  class_dict=spec.__dict__['class_dict'],
                                                  val_fold=spec.__dict__['val_fold'],
                                                  batch_size=spec.__dict__['batch_size'])
    callbacks = create_callbacks(spec.get_id())
    if 'fit_verbose' in pacltune.conf.keys():
        if pacltune.conf['fit_verbose']:
            verbose = 1
        else:
            verbose = 2
    else:
        verbose = 2
    print("Start tuning.")
    if len(pacltune.physical_gpus) > 0:
        r_mon=pacltune.rmon.ResMonitor(1)
    history = model.fit(x=tfdata_train,
                        validation_data=tfdata_val,
                        callbacks=callbacks,
                        verbose=verbose,
                        epochs=spec.__dict__['n_epochs'],
                        initial_epoch=initial_epoch)
    if len(pacltune.physical_gpus) > 0:
        r_mon.exit()

def run():
    pacltune.utils.clean() # MIght casue problems when running 'run' two times in a row
    if not os.path.isfile('input/specifications.py'):
        raise Exception('File \'input/specifications.py\' missing.')
    spec = importlib.util.spec_from_file_location("mod", "input/specifications.py")
    spec_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(spec_mod)
    spec_dicts = spec_mod.spec_dicts
    if type(spec_dicts).__name__ == 'dict':
        spec_dicts = [spec_dicts]
    # Iterate across specs
    n_specs = len(spec_dicts)
    for i in range(n_specs):
        tf.keras.backend.clear_session()
        print('### Specification ' + str(i + 1) + '\\' + str(n_specs) + '.')
        print(spec_dicts[i])
        spec = pacltune.model.Specification(spec_dicts[i])
        print(spec.get_id())
        fitpacl(spec)
        print('')
