from detector import *

import argparse
from unicodedata import name
import time


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-mp', '--model_path', type=str,
                        default=None,
                        help='path to already trained model, only relevant in when re-training model')


    parser.add_argument('-ujp', '--url_json_path', type=str,
                        default=None,
                        help='path to urls to predict')
                        

    args = parser.parse_args()

    assert args.model_path is not None, 'Need to provide path to already trained model to use in prediction mode'
    detector = Pinner( 
        model_path = args.model_path, 
    )

    pred_dict = detector.predict_transform(
        url_json_path = args.url_json_path,
        return_boxes = True
    )

    return pred_dict

if __name__ == '__main__':
    print(main())