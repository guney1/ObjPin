import json
import os
import re
import shutil
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from ultralytics import YOLO
import requests
from datetime import datetime

@dataclass
class Pinner:
    run_id: str = None
    exp_id: str = None
    model = None
    model_path: str = None
    train_config_path: str = None

    def __post_init__(self):
        if self.model_path is not None:
            self.is_fit = True
        else:
            self.is_fit = False

        if self.run_id is None:
            self.run_id = datetime.utcnow().strftime("%Y%m%d_%H%M")
        if self.exp_id is None:
            self.exp_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        self.project_name = f'{self.run_id}_project'
        self.experiment_name = f'{self.exp_id}_exp'



    def generate_training_folders(self, pre_folder_path, traning_folder_path, test_size=0.2):
        assert 'images' in os.listdir(pre_folder_path), f'There is no image folder inside {pre_folder_path}'
        assert 'labels' in os.listdir(pre_folder_path), f'There is no image folder inside {pre_folder_path}'
        
        img_root = f'{pre_folder_path}/images'
        label_root = f'{pre_folder_path}/labels'
        
        image_file_names = [re.sub('\.\w+', '', img_name) for img_name in os.listdir(img_root)]
        label_file_names = [re.sub('\.\w+', '', label_name) for label_name in os.listdir(label_root)]

        final_files = list(set(image_file_names) & set(label_file_names))

        image_file_names = [img_name + '.jpg' for img_name in final_files]
        label_file_names = [img_name + '.txt' for img_name in final_files]

        df_mapper = pd.DataFrame()
        df_mapper['images'] = image_file_names
        df_mapper['labels'] = label_file_names

        test_abs_size = round(len(df_mapper) * test_size)
        test_idx = np.random.choice(df_mapper.index, test_abs_size, replace=False)
        train_idx = np.array([a for a in df_mapper.index if a not in test_idx])

        df_train = df_mapper.loc[train_idx, :]
        df_test = df_mapper.loc[test_idx, :]        
        
        if not os.path.exists(f'./{traning_folder_path}'):
            os.makedirs(f'{traning_folder_path}/images/train')
            os.makedirs(f'{traning_folder_path}/labels/train')

            os.makedirs(f'{traning_folder_path}/images/test')
            os.makedirs(f'{traning_folder_path}/labels/test')

        for row in df_train.values:
            img_name, label_name = row
            shutil.move(f'{img_root}/{img_name}', f'{traning_folder_path}/images/train/{img_name}')
            shutil.move(f'{label_root}/{label_name}', f'{traning_folder_path}/labels/train/{label_name}')

        for row in df_test.values:
            img_name, label_name = row
            shutil.move(f'{img_root}/{img_name}', f'{traning_folder_path}/images/test/{img_name}')
            shutil.move(f'{label_root}/{label_name}', f'{traning_folder_path}/labels/test/{label_name}')

    def _fit(self, epochs, train_config):
        if self.is_fit:
            self._load_model()
        else:
            self.model = YOLO(model="yolov8n.yaml")
        
        self.model.train(data=train_config, epochs=epochs, project=self.project_name, name=self.experiment_name, resume=False)

    def _load_model(self):
        self.model = YOLO(self.model_path)

    def _get_target_images(self, url_json_path):
        self.url_mapper = {}
        os.mkdir('./prediction_folder')
        with open(url_json_path) as f:
            content = f.read()
        url_dict = json.loads(content)
        url_list = url_dict['url_list']
        for n, url in enumerate(url_list):
            try:
                img_data = requests.get(url).content
                self.url_mapper[f'{n}.jpg'] = url
                with open(f'./prediction_folder/{n}.jpg', 'wb') as handler:
                    handler.write(img_data)
            except:
                pass        

    def _predict(self, url_json_path):
        self._get_target_images(url_json_path)
        img_names = os.listdir('./prediction_folder')
        img_paths = ['./prediction_folder/' + file_name for file_name in img_names]
        preds = self.model(img_paths)
        pred_dict = {}
        for n in range(len(preds)):
            detect_boxes = list(preds[n].boxes.xyxy)
            n_detect = len(detect_boxes)
            pred_dict[img_names[n]] = {'boxes': detect_boxes, 'n_detection': n_detect}

        shutil.rmtree('./prediction_folder')
        return pred_dict

    def fit_transform(self, train_config_path, epochs=500):
        self._fit(epochs=epochs, train_config=train_config_path)

    def predict_transform(self, url_json_path, return_boxes=False):
        self._load_model()
        pred_dict = self._predict(url_json_path)
        if not return_boxes:
            pred_dict = {key: pred_dict[key]['n_detection'] for key in pred_dict}

        pred_dict_final = {self.url_mapper[key]: pred_dict[key] for key in pred_dict}


        return pred_dict_final

