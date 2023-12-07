# Setup the environment

 - Create your virtual env
 - Install the requirements for the module through `requirements.txt`


# Train mode

 - Corresponding file: `main_train.py`

## Pre-requisite flags

 - `--sub_training_directory`: This folder should contain YOLO formatted images and labels
 - `--training_directory`: Necessary folder structure will be generated with this name and files from above dir will move here based on train-test split
 - `--train_config_path`: This is necessary yaml file to train the model:
    - path should be the full path of --training_directory
    - train and val keys set it same as the example in train_config.yaml
    - names key, leave the 0 there string part can be anything

## All possible flags (above flags are mandatory not mentioned in this section)

 - `--unique_run_id`: if not provided will be generated auto
 - `--unique_exp_id`: if not provided will be generated auto
 - `--model_path`: only relevant when we want to train already trained model

## How to run

Simplest way to run it sufficient

 - `python main_train.py -stdir ./sub_train_folder -tdir ./train_folder -tcp ./train_config.yaml`

Note that `-tdir` path should match the path key provided in `-tcp`.

To be safe, provide full path to every key in flags

# Predict mode

 - Corresponding file: `main_predict.py`

## Pre-requisite flags

 - `--model_path`: Path to already trained model
 - `--url_json_pat`h: Path to json file that contains urls we want to make a prediction

## How to run

Once model trained and saved can make predictions on images, note that urls should be valid. Downloaded images will be deleted once the predictions returned

 - `python main_predict.py -mp <PATH TO ROOT OF best.pt>/best.pt -ujp <PATH TO JSON FILE THAT CONTAINS URLs>`
