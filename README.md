# PaclTune

A python package for the tuning of image **pa**tch based **cl**assifiers in medical applications.

## Usage

The main function of PaclTune is the following.

```python
pacltune.tune.run()
```

If this is run from the project directory (which must be set up with the required inputs; see below) all models from the given specification will be tuned.

Once some models are tuned, predictions are requested by the following command.

```python
pacltune.predict.run()
```

This will create a prediction table (for each model and set requested) in the folder `output/predictions`

## Inputs

### CSV files for image patches

These files are `callibration-patches.csv` and `test-patches.csv` in the project structure at the bottom of this README.

#### 1) Callibration

A csv file containing meta information on the image patches for training and validation:
  1. **file:** The filename of the image patch (without the directory). E.g., 'image1.jpg'.
  2. **pacl_class:** The class the patch belongs to. E.g., 'tumor-type1'.
  3. **fold:** Integer (>= 0) indicating the fold used for validation. It defaults to the first (fold == 0) and can be changed in the model specifications (see below). Thus, by default image patches with fold == 0 will be used for validation, while all others in `callibration-patches.csv` will be used for training.

#### 2) Test

A csv file containing meta information on the test image patches. It needs the same attributes as the callibration file except *fold*.

### JSON file coding classes to integers

A JSON file with a dictionary that maps the classes (from `pacl_class` in `callibration-patches.csv` or `test-patches.csv`) to integers (>= 0). This file is `class-dict.json` in the project structure at the bottom of this README. For example, if there is two classes `c1` and `c2` the dictionary in `class-dict.json` might be:

```python
{
  "c1" : 0,
  "c2" : 1
}
```

### Model specification

A python file with a dictionary giving any parameter that deviates from the default values (an empty dictionary will work as well). To see all available parameters and respective default values use `print(pacltune.defaults)`. The exemplary dictionary below specifies an Xception model and learning rate different from the defaults.

```python
spec_dicts = {
  "app" : "Xception",
  "lr" : 1e-02
}
```

See [here](https://www.tensorflow.org/api_docs/python/tf/keras/applications?hl=de) for available applications (`app`) and [here](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers) for optimizers (`optimizer`).

Multiple different model specifications can be supplied as a list. This can be very useful for testing a range of parameter values (e.g. grid based search). The code below creates model specifications with a variety of learning rates.

```python
lrs = [1e-02, 1e-03, 1e-04]
apps = ["Xception", "EfficientNetB4"]
spec_dicts = [{"app" : app, "lr" : lr, "drop" : drop, "augment" : True} for lr in lrs for app in apps]
```

The python dictionaries should be written to the `specifications.py` file (see the project structure below). This file will be read for a dictionary or list of name `spec_dicts` when running: `pacltune.tune.run()`.

**IMPORTANT:** Since `from input.specifications import spec_dicts` is used to read `spec_dicts` a few things need to be taken care of:

- There should be a single dictionary or list called `spec_dicts` in `input/specifications.py`.
- When calling `pacltune.tune.run()` from a python script not within the project directory the `sys.path` variable must be adopted such that the module `input.specifications` can be found. For example, run `sys.path.insert(0,'')` to add the current working directory at the start of `sys.path`.
- A modification of `input/specifications.py` will only take affect when the python session is restarted.

### Model prediction

A python file with a dictionary giving the model id and set (i.e., 'train', 'val' or 'test') that are to be predicted. Similar to the dictionary for specifications, it is supplied as a list (named `pred_dicts`) of dictionaries in the file `input/to_predict.py`. Every dictionary in the list should have the key `id` and `set` with a unique value each. Here an example for predictions on all sets of a single model.

```python
sets = ["train", "val", "test"]
pred_dicts = [{"id" : "1ec218cb7f2bf1a74087e1dd31afe34e", "set" : set} for set in sets]
```

### Project Specification

A YAML file that sets up project details (like the path to the image patches). this file might look like this:

```YAML
default:
  path_to_patches: ../Patches
  data_version: '1.0'
  fit_verbose: no
```

## Avoid duplicate tuning

Next to implementing the tuning of a [TensorFlow](https://www.tensorflow.org/) model, PaclTune also tracks models that have already been tuned and avoids (unintentional) duplication. PaclTune achieves this by hashing the model specification dictionary (this is the models ID). All parameters but the number of epochs is included in the hash. The number of epochs are excluded since the training of models can continue (which clearly changes the number of epochs). The data version (specified in project.yml) is included in the hash since its part of the default model specification (and all default values are included). This applies to the contents of `input/class_dict.json` as well.

Sometimes it might be of interest to duplicate a model intentionally. For example, to account for the randomness involved in training models (e.g. in the weights initialization). For this reason the parameter `rep` can be set (an integer that defaults to 0).

## Project structure

```
├── project.yml
├── input
│   ├── callibration-patches.csv
│   ├── class-dict.json
│   ├── to_predict.py
│   └── specifications.py
└── output
    ├── data
    │   └── test-patches.csv
    ├── models
    │   └── 1ec218cb7f2bf1a74087e1dd31afe34e
    │       ├── history.csv
    │       ├── spec.json
    │       └── latest_model.h5
    └── predictions
        └── val_1ec218cb7f2bf1a74087e1dd31afe34e.csv
```

Copyright (C) 2021 University Hospital Heidelberg
