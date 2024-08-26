import os, shutil
import glob
import logging
import yaml
from datetime import datetime

from ultralytics import YOLO
from PIL import Image

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size, is_skipped
from label_studio.core.utils.io import get_data_dir

logger = logging.getLogger(__name__)

IMG_DATA = os.path.join(os.path.dirname(__file__), 'data', 'images')
LABEL_DATA = os.path.join(os.path.dirname(__file__), 'data', 'labels')
INIT_WEIGHTS = os.path.join(os.path.dirname(__file__), 'config', 'checkpoints', 'starting_weights.pt') #save location for finetuned weights
TRAINED_WEIGHTS = os.path.join(os.path.dirname(__file__), 'config', 'checkpoints', 'trained_weights.pt') #save location for weights after training
CONFIG = os.path.join(os.path.dirname(__file__), 'config', 'data.yaml')

class MosaicItemDetector(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(MosaicItemDetector, self).__init__(**kwargs)
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')

        self.image_dir = upload_dir
        logger.debug(
            f'{self.__class__.__name__}  reads image from {self.image_dir}')

        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema['to_name'][0]
        self.labels = schema['labels']
        self.class_totals = {label : 0 for label in self.labels}
        # print(self.class_totals)
        print(CONFIG)
        with open(CONFIG, 'r') as file:
            self.prime_service = yaml.safe_load(file)

        if os.path.isfile(TRAINED_WEIGHTS):
            self.weights = TRAINED_WEIGHTS
        else:
            self.weights = INIT_WEIGHTS

        self.model = YOLO(self.weights)
        #self.model = YOLO('yolov8x.yaml')

    def predict(self, tasks, **kwargs):
        #print('predict')
        now = datetime.now()
        modelname = "yolov8x" + now.strftime("%H:%M:%S")
        predictions = []
        for task in tasks:
            lowest_conf = 2.0
            img_results = []
            #print('task')
            image_url = task['data']['image']
            image_path = self.get_local_path(
                image_url, project_dir=self.image_dir)
            img = Image.open(image_path)
            img_w, img_h = get_image_size(image_path)

            objs = self.model.predict(img)
            for obj in objs:
                for box in obj.boxes:
                    # print(obj)
                    # print(box)
                    # print(box.xywh)
                    x, y, w, h = box.xywh[0]
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x = 100 * float(x - w / 2) / img_w
                    y = 100 * float(y - h / 2) / img_h
                    w = 100 * float(w) / img_w
                    h = 100 * float(h) / img_h
                    if conf < lowest_conf:
                        lowest_conf = conf
                    label = self.labels[cls]

                    img_results.append({
                        "from_name": self.from_name,
                        'to_name': self.to_name,
                        "original_width": img_w,
                        "original_height": img_h,
                        'type': 'rectanglelabels',
                        'value': {
                            'rectanglelabels': [label],
                            'x': x,
                            'y': y,
                            'width': w,
                            'height': h,
                        },
                        'score': conf
                    })
            score = 1.0
            if lowest_conf <= 1.0:
                score = lowest_conf
            predictions.append({
                "score": score,
                'model_version': modelname,
                'result': img_results,
            })

        return predictions
    
    def label2idx(self, label):
        return self.prime_service['names'].index(label)

    def move_files(self, files, label_img_data):
        #move files to train or val directories
        # print(files)
        print("moving files")
        
        for ix, file in enumerate(files):
            file_name = os.path.basename(file)

            train_val = "val/"
            if file_name[0:3] == 'trn':
                train_val = "train/"

            dest = os.path.join(label_img_data,train_val,file_name)
            # print(dest)
            shutil.move(file, dest)
    
    def reset_train_dir(self, dir_path):
        #remove cache file and reset train/val dir
        if os.path.isfile(os.path.join(dir_path,"train.cache")):
            os.remove(os.path.join(LABEL_DATA, "train.cache"))
            os.remove(os.path.join(LABEL_DATA, "val.cache"))

        for dir in os.listdir(dir_path):
            shutil.rmtree(os.path.join(dir_path, dir))
            os.makedirs(os.path.join(dir_path, dir))
    
    def extract_data_from_tasks(self, tasks):
        img_labels = []
        dest_codes = {'train': 'trn', 'valid': 'val'}
        for task in tasks:
            if is_skipped(task):
                continue
            # print(task)
            image_url = task['data']['image']
            split_dest = dest_codes[task['data']['split']]

            image_path = self.get_local_path(image_url)
            image_name = image_path.split("\\")[-1]
            Image.open(image_path).save(os.path.join(IMG_DATA, split_dest + image_name))

            img_labels.append(task['annotations'][0]['result'])

            for annotation in task['annotations']:
                for bbox in annotation['result']:
                    bb_width = (bbox['value']['width']) / 100
                    bb_height = (bbox['value']['height']) / 100
                    x = (bbox['value']['x'] / 100 ) + (bb_width/2)
                    y = (bbox['value']['y'] / 100 ) + (bb_height/2)
                    label = bbox['value']['rectanglelabels']
                    # print(label[0])
                    # print(self.class_totals[label[0]])
                    self.class_totals[label[0]] = self.class_totals[label[0]] + 1
                    label_idx = self.label2idx(label[0])
                        
                    with open(os.path.join(LABEL_DATA, split_dest + image_name[:-4]+'.txt'), 'a') as f:
                        f.write(f"{label_idx} {x} {y} {bb_width} {bb_height}\n")
        
        for c, num in self.class_totals.items():
            # print(self.class_totals)
            # print(c)
            print(c, ':', num)
    
    def fit(self, tasks, workdir=None, **kwargs):
        print('Init Training')
        for dir_path in [IMG_DATA, LABEL_DATA]:
            print(dir_path)
            self.reset_train_dir(dir_path)
        
        self.extract_data_from_tasks(tasks)

        img_files = glob.glob(os.path.join(IMG_DATA, "*.jpg"))
        label_files = glob.glob(os.path.join(LABEL_DATA, "*.txt"))

        self.move_files(img_files, IMG_DATA)
        self.move_files(label_files, LABEL_DATA)

        print("Start training")
        #self.model.train(exist_ok=True, epochs=100, data=CONFIG, batch=-1, imgsz=640, device=0, name='mosaic-training')
        self.model.train(exist_ok=True, epochs=100, data=CONFIG, batch=-1, model=self.weights, imgsz=640, device=0, name='mosaic-training')
        
        shutil.move(os.path.join(os.path.dirname(__file__), 'runs', 'detect', 'mosaic-training', 'weights', 'best.pt'), TRAINED_WEIGHTS)#move trained weights to checkpoint folder
        print("done training")

        self.weights = TRAINED_WEIGHTS #updating to trained weights
        print(f"The new weights are: {self.weights}")
            
        return {
            'model_path': TRAINED_WEIGHTS,
        }
