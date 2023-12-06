from detector import *

import argparse
from unicodedata import name
import time


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-stdir', '--sub_training_directory', type=str,
                        default=None,
                        help='name of the dir, original images stored annonated images stored in correct format')

    parser.add_argument('-tdir', '--training_directory', type=str,
                        default=None,
                        help='name of the dir, training file structure will be stored')

    parser.add_argument('-rid', '--unique_run_id', type=str,
                        default=None,
                        help='Unique id, model files will be stored with same name')

    parser.add_argument('-xid', '--unique_exp_id', type=str,
                        default=None,
                        help='Unique id, model files will be stored with same name under run id folder')

    parser.add_argument('-mp', '--model_path', type=str,
                        default=None,
                        help='path to already trained model, only relevant in when re-training model')


    parser.add_argument('-tcp', '--train_config_path', type=str,
                        default=None,
                        help='Path to training config')
                        

    args = parser.parse_args()

    detector = Pinner(
        run_id = args.unique_run_id, 
        exp_id = args.unique_exp_id, 
        model_path = args.model_path, 
        train_config_path = args.train_config_path
    )

    detector.generate_training_folders(
        pre_folder_path = args.sub_training_directory, 
        traning_folder_path = args.training_directory
    )

    detector.fit_transform(
        train_config_path=args.train_config_path, 
        epochs=5
    )



if __name__ == '__main__':
    main()