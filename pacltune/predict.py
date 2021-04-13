
import pacltune
import os
import pandas as pd
import tensorflow as tf
from datetime import datetime
import importlib

def run():
    if not os.path.isfile('input/to_predict.py'):
        raise Exception('File \'input/to_predict.py\' missing.')
    spec = importlib.util.spec_from_file_location("mod", "input/to_predict.py")
    spec_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(spec_mod)
    pred_dicts = spec_mod.pred_dicts
    if type(pred_dicts).__name__ == 'dict':
        pred_dicts = [pred_dicts]
    # Iterate across specs
    n_preds = len(pred_dicts)
    for i in range(n_preds):
        tf.keras.backend.clear_session()
        print('### Prediction ' + str(i + 1) + '\\' + str(n_preds) + '.')
        by_ids(ids = [pred_dicts[i]['id']], set=pred_dicts[i]['set'])
        print('')

# set is either 'val' or 'test'
def by_ids(ids, set='val'):
    for id in ids:
        print('# Model ' + id + ' on set ' + set + '.')
        model, spec = pacltune.model.model_from_id_for_prediction(id)
        create_prediction(model=model, spec=spec, set=set)

def create_prediction(model, spec, set, verbose=2):
    input_shape = model.get_layer(index=0).input.get_shape().as_list()
    tf_data, patches_tab = pacltune.data.init_eval(set=set, input_size=input_shape[1], val_fold=spec.__dict__['val_fold'], batch_size=1)
    predictions = model.predict(x=tf_data, verbose=verbose)
    patches_tab['pred_class'] = tf.math.argmax(predictions, axis=1)
    inv_class_dict = {v: k for k, v in spec.__dict__['class_dict'].items()}
    [inv_class_dict[i] for i in patches_tab['pred_class']]
    patches_tab['pred_class'] = [inv_class_dict[i] for i in patches_tab['pred_class']]
    patches_tab['pred_prob'] = tf.math.reduce_max(predictions, axis=1)
    now = datetime.now()
    if not os.path.isdir('output/predictions'):
        os.makedirs('output/predictions')
    patches_tab.to_csv('output/predictions/' + set + '_' + now.strftime("%Y%m%d") + '_' + spec.get_id() + '.csv', index=False)
