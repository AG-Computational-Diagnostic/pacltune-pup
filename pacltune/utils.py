
import yaml
import socket
import os
import json
import shutil
import pandas as pd

def read_yaml_to_dict(yml_file):
    with open(yml_file, 'r') as stream:
        try:
            nested_dict = yaml.safe_load(stream)
            dict = nested_dict['default']
        except yaml.YAMLError as exc:
            print(exc)
    if socket.gethostname().lower() in nested_dict:
        dict.update(nested_dict[socket.gethostname().lower()])
    return(dict)

def read_prj_conf():
    print('# Project config for ' + socket.gethostname().lower() + ':')
    if os.path.isfile('project.yml'):
        conf = read_yaml_to_dict('project.yml')
        {print(x + ': ' + str(conf[x])) for x in conf}
    else:
        conf = {}
        print('No config found (project.yml)')
    return(conf)

def read_json_dict(class_dict_file):
    class_dict = json.load(open(class_dict_file))
    return class_dict

def read_class_dict_with_warn(class_dict_name):
    class_dict_file = 'input/' + class_dict_name + '.json'
    if os.path.isfile(class_dict_file):
        class_dict = read_json_dict(class_dict_file)
    else:
        print('WARNING:pacltune:\'' + class_dict_file + '\' does not exist.')
        class_dict = {}
    return class_dict

def get_spec_dict(model_id):
    spec_dict_file = get_model_path(model_id) + '/' + 'spec.json'
    if os.path.isfile(spec_dict_file):
        spec_dict = read_json_dict(spec_dict_file)
    else:
        raise Exception('Model with id ' + model_id + ' has no spec.json.')
    return spec_dict

def get_models_dir():
    return 'output/models/'

def get_model_path(model_id):
    model_path = get_models_dir() + model_id
    return model_path

def get_model_file(model_id):
    return get_model_path(model_id) + '/latest_model.h5'

def get_history_file(model_id):
    return get_model_path(model_id) + '/history.csv'

def get_spec_file(model_id):
    return get_model_path(model_id) + '/spec.json'

def purge(model_id):
    print("Moving model " + model_id + " to \'output/trash/\'.")
    if not os.path.isdir('output/trash'):
        os.makedirs('output/trash')
    shutil.move(get_model_path(model_id), 'output/trash/')

def clean():
    if os.path.isdir(get_models_dir()):
        model_ids = os.listdir(get_models_dir())
        for model_id in model_ids:
            model_files = os.listdir(get_model_path(model_id))
            nec_files = ['history.csv', 'latest_model.h5', 'spec.json']
            if not all(f in model_files for f in nec_files):
                print('Model ' + model_id + ' does not have all necessary files and will be purged. See existing files below.')
                print(model_files)
                purge(model_id)
