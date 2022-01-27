# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from tqdm import tqdm
import sys
import numpy as np
from tqdm.notebook import tqdm
tqdm.pandas()
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import glob
import shutil
import sys
sys.path.append('../input/tensorflow-great-barrier-reef')
import torch
from PIL import Image
import ast
sys.path.append('../input/tensorflow-great-barrier-reef')

# %%
!cp -r /kaggle/input/bytetrack /kaggle/working/tmp/

# %%


# %%
!pip install /kaggle/working/tmp/cython_bbox-0.1.3/cython_bbox-0.1.3
!pip install /kaggle/working/tmp/lap-0.4.0/lap-0.4.0
!pip install /kaggle/working/tmp/loguru-0.5.3-py3-none-any.whl
!pip install /kaggle/working/tmp/ninja-1.10.2.2-py2.py3-none-manylinux_2_5_x86_64.manylinux1_x86_64.whl
!pip install /kaggle/working/tmp/thop-0.0.31.post2005241907-py3-none-any.whl
!pip install /kaggle/working/tmp/pycocotools-2.0.2/dist/pycocotools-2.0.2.tar

!pip install /kaggle/working/tmp/onnx-1.8.0-cp37-cp37m-manylinux2010_x86_64.whl
!pip install /kaggle/working/tmp/onnxoptimizer-0.2.6-cp37-cp37m-manylinux2014_x86_64.whl
!pip install /kaggle/working/tmp/onnx-simplifier-0.3.5/onnx-simplifier-0.3.5

!pip install /kaggle/working/tmp/pycocotools-2.0.2/dist/pycocotools-2.0.2.tar


%cd /kaggle/working
!cp -r ../input/bytetrack/ByteTrack /kaggle/working/
%cd /kaggle/working/ByteTrack
!pip install -e . --no-deps
%cd /kaggle/working/

# %%
sys.path.append('../input/bytetrack/ByteTrack')

# %%
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

# %%
CONF      = 0.01
IOU       = 0.20
IMG_SIZE  = 3600
AUGMENT   = True
CKPT_PATH = '../input/reef-baseline-fold12/l6_3600_uflip_vm5_f12_up/f1/best.pt'

# %%
def voc2yolo(bboxes, image_height=720, image_width=1280):
    """
    voc  => [x1, y1, x2, y1]
    yolo => [xmid, ymid, w, h] (normalized)
    """
    
    bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]]/ image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]]/ image_height
    
    w = bboxes[..., 2] - bboxes[..., 0]
    h = bboxes[..., 3] - bboxes[..., 1]
    
    bboxes[..., 0] = bboxes[..., 0] + w/2
    bboxes[..., 1] = bboxes[..., 1] + h/2
    bboxes[..., 2] = w
    bboxes[..., 3] = h
    
    return bboxes

def yolo2voc(bboxes, image_height=720, image_width=1280):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    voc  => [x1, y1, x2, y1]
    
    """ 
    bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]]* image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]]* image_height
    
    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]]/2
    bboxes[..., [2, 3]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]]
    
    return bboxes

def coco2yolo(bboxes, image_height=720, image_width=1280):
    """
    coco => [xmin, ymin, w, h]
    yolo => [xmid, ymid, w, h] (normalized)
    """
    
    bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    
    # normolizinig
    bboxes[..., [0, 2]]= bboxes[..., [0, 2]]/ image_width
    bboxes[..., [1, 3]]= bboxes[..., [1, 3]]/ image_height
    
    # converstion (xmin, ymin) => (xmid, ymid)
    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]]/2
    
    return bboxes

def yolo2coco(bboxes, image_height=720, image_width=1280):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    coco => [xmin, ymin, w, h]
    
    """ 
    bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    
    # denormalizing
    bboxes[..., [0, 2]]= bboxes[..., [0, 2]]* image_width
    bboxes[..., [1, 3]]= bboxes[..., [1, 3]]* image_height
    
    # converstion (xmid, ymid) => (xmin, ymin) 
    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]]/2
    
    return bboxes

def voc2coco(bboxes, image_height=720, image_width=1280):
    bboxes  = voc2yolo(bboxes, image_height, image_width)
    bboxes  = yolo2coco(bboxes, image_height, image_width)
    return bboxes


def load_image(image_path):
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def draw_bboxes(img, bboxes, classes, class_ids, colors = None, show_classes = None, bbox_format = 'yolo', class_name = False, line_thickness = 2):  
     
    image = img.copy()
    show_classes = classes if show_classes is None else show_classes
    colors = (0, 255 ,0) if colors is None else colors
    
    if bbox_format == 'yolo':
        
        for idx in range(len(bboxes)):  
            
            bbox  = bboxes[idx]
            cls   = classes[idx]
            cls_id = class_ids[idx]
            color = colors[cls_id] if type(colors) is list else colors
            
            if cls in show_classes:
            
                x1 = round(float(bbox[0])*image.shape[1])
                y1 = round(float(bbox[1])*image.shape[0])
                w  = round(float(bbox[2])*image.shape[1]/2) #w/2 
                h  = round(float(bbox[3])*image.shape[0]/2)

                voc_bbox = (x1-w, y1-h, x1+w, y1+h)
                plot_one_box(voc_bbox, 
                             image,
                             color = color,
                             label = cls if class_name else str(get_label(cls)),
                             line_thickness = line_thickness)
            
    elif bbox_format == 'coco':
        
        for idx in range(len(bboxes)):  
            
            bbox  = bboxes[idx]
            cls   = classes[idx]
            cls_id = class_ids[idx]
            color = colors[cls_id] if type(colors) is list else colors
            
            if cls in show_classes:            
                x1 = int(round(bbox[0]))
                y1 = int(round(bbox[1]))
                w  = int(round(bbox[2]))
                h  = int(round(bbox[3]))

                voc_bbox = (x1, y1, x1+w, y1+h)
                plot_one_box(voc_bbox, 
                             image,
                             color = color,
                             label = cls if class_name else str(cls_id),
                             line_thickness = line_thickness)

    elif bbox_format == 'voc_pascal':
        
        for idx in range(len(bboxes)):  
            
            bbox  = bboxes[idx]
            cls   = classes[idx]
            cls_id = class_ids[idx]
            color = colors[cls_id] if type(colors) is list else colors
            
            if cls in show_classes: 
                x1 = int(round(bbox[0]))
                y1 = int(round(bbox[1]))
                x2 = int(round(bbox[2]))
                y2 = int(round(bbox[3]))
                voc_bbox = (x1, y1, x2, y2)
                plot_one_box(voc_bbox, 
                             image,
                             color = color,
                             label = cls if class_name else str(cls_id),
                             line_thickness = line_thickness)
    else:
        raise ValueError('wrong bbox format')

    return image

def get_bbox(annots):
    bboxes = [list(annot.values()) for annot in annots]
    return bboxes

def get_imgsize(row):
    row['width'], row['height'] = imagesize.get(row['image_path'])
    return row

np.random.seed(32)
colors = [(np.random.randint(255), np.random.randint(255), np.random.randint(255))\
          for idx in range(1)]

# %% [markdown]
# 

# %%
def get_path(row):
    row['image_path'] = f'{ROOT_DIR}/train_images/video_{row.video_id}/{row.video_frame}.jpg'
    return row

# %%
ROOT_DIR  = '/kaggle/input/tensorflow-great-barrier-reef/'
# Train Data
df = pd.read_csv(f'{ROOT_DIR}/train.csv')
df = df.progress_apply(get_path, axis=1)
df['annotations'] = df['annotations'].progress_apply(lambda x: ast.literal_eval(x))
display(df.head(2))

# %%
df['num_bbox'] = df['annotations'].progress_apply(lambda x: len(x))
data = (df.num_bbox>0).value_counts()/len(df)*100
print(f"No BBox: {data[0]:0.2f}% | With BBox: {data[1]:0.2f}%")

# %%
!mkdir -p /root/.config/Ultralytics
!cp /kaggle/input/yolov5-font/Arial.ttf /root/.config/Ultralytics/

# %%
def load_model(ckpt_path, conf=0.01, iou=0.50):
    model = torch.hub.load('../input/yolov5-lib-ds',
                           'custom',
                           path=ckpt_path,
                           source='local',
                           force_reload=True)  # local repo
    model.conf = conf  # NMS confidence threshold
    model.iou  = iou  # NMS IoU threshold
    model.classes = None   # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image
    return model

# %%
def predict(model, img, size=768, augment=False):
    height, width = img.shape[:2]
    results = model(img, size=size, augment=augment)  # custom inference size
    preds   = []
    detects = []
    if results.pandas().xyxy[0].shape[0] == 0:
        return [],[]
    else:
        for idx, row in results.pandas().xyxy[0].iterrows():
            if row.confidence > 0.01:
                detects.append([int(row.xmin), int(row.ymin), int(row.xmax), int(row.ymax), row.confidence])
                preds.append([row.xmin, row.ymin, row.xmax-row.xmin, row.ymax-row.ymin])
                #     bboxes  = preds[['xmin','ymin','xmax','ymax', 'confidence']].values.astype(int)
#         bboxes  = voc2coco(bboxes,height,width).astype(int)
#         confs   = preds.confidence.values
        return detects, preds
        
    
def format_prediction(bboxes, confs):
    annot = ''
    if len(bboxes)>0:
        for idx in range(len(bboxes)):
            xmin, ymin, w, h = bboxes[idx]
            conf             = confs[idx]
            annot += f'{conf} {xmin} {ymin} {w} {h}'
            annot +=' '
        annot = annot.strip(' ')
    return annot

def show_img(img, bboxes, bbox_format='yolo'):
    names  = ['starfish']*len(bboxes)
    labels = [0]*len(bboxes)
    img    = draw_bboxes(img = img,
                           bboxes = bboxes, 
                           classes = names,
                           class_ids = labels,
                           class_name = True, 
                           colors = colors, 
                           bbox_format = bbox_format,
                           line_thickness = 2)
    return Image.fromarray(img).resize((800, 400))

# %%
%cd /kaggle/working

# %%
model = load_model(CKPT_PATH, conf=CONF, iou=IOU)
# image_paths = df[df.num_bbox>1].sample(100).image_path.tolist()
image_paths = df[df.sequence == 53708].image_path.tolist()
for idx, path in enumerate(image_paths):
#     break
    img = cv2.imread(path)[...,::-1]
    detects, preds = predict(model, img, size=IMG_SIZE, augment=AUGMENT)
    print(len(preds))
#     display(show_img(img, preds, bbox_format='coco'))
    if idx>15:
        break

# %%
#######################################################
#                      Tracking                       #
#######################################################

# Tracker will update tracks based on detections from current frame
# Matching based on euclidean distance between bbox centers of detections 
# from current frame and tracked_objects based on previous frames
# You can check it's parameters in norfair docs
# https://github.com/tryolabs/norfair/blob/master/docs/README.md
# tracker = Tracker(
#     distance_function=euclidean_distance, 
#     distance_threshold=30,
#     hit_inertia_min=3,
#     hit_inertia_max=6,
#     initialization_delay=1,
# )
class args:
    det_thresh = 0.011
    track_thresh = 0.001
    track_buffer = 30
    mot20 = False
    match_thresh = 0.001
#     aspect_ratio_thresh = 1.6
    min_box_area = 1000
    
from yolox.tracker.byte_tracker import BYTETracker
tracker = BYTETracker(args)

# Save frame_id into detection to know which tracks have no detections on current frame
frame_id = 0
#######################################################

# %%
model = load_model(CKPT_PATH, conf=CONF, iou=IOU)
# image_paths = df[df.num_bbox>1].sample(100).image_path.tolist()
# image_paths = df[100:200].image_path.tolist()
image_paths = df[df.sequence == 53708].image_path.tolist()
for idx, path in enumerate(image_paths):
    img = cv2.imread(path)[...,::-1]
    height, width = img.shape[:2]
    detects, preds = predict(model, img, size=IMG_SIZE, augment=AUGMENT)
#     r = model(img, size=10000, augment=False)
    #######################################################
    #                      Tracking                       #
    #######################################################
    preds = [] # if you want to visualize bboxes detected with bytetrack(bytetrackでの追跡のみ表示したくなければコメントアウトする)
    # Update tracks using detects from current frame
    if len(detects):
        tracked_objects = tracker.update(np.array(detects), [height, width], np.array([IMG_SIZE, IMG_SIZE]))
#         print(len(tracked_objects))
#         print("detects: {}, tracked_object: {}".format(len(detects), len(tracked_objects)))
        for tobj in tracked_objects:
            # Add objects that have no detections on current frame to predictions
            tlwh = tobj.tlwh
            if tlwh[2] * tlwh[3] > args.min_box_area:
                x_min = int(tlwh[0])
                y_min = int(tlwh[1])
                bbox_width = int(tlwh[2])
                bbox_height = int(tlwh[3])
                preds.append([x_min, y_min, bbox_width, bbox_height])
                score = tobj.score
                print('{} {:.2f} {} {} {} {}'.format(frame_id, score, x_min, y_min, bbox_width, bbox_height))
    #         preds.append('{:.2f} {} {} {} {}'.format(score, x_min, y_min, bbox_width, bbox_height))
        #######################################################
                display(show_img(img, preds, bbox_format='coco'))
    if idx>100:
        break
    frame_id +=1

# %%
detects

# %%
tracked_objects = tracker.update(np.array(detects), [height, width], np.array([IMG_SIZE, IMG_SIZE]))

# %%
tracker.info

# %%
import greatbarrierreef
env = greatbarrierreef.make_env()# initialize the environment
iter_test = env.iter_test()      # an iterator which loops over the test set and sample submission

# %%


# %%
model = torch.hub.load('../input/yolov5-lib-ds', 
                       'custom', 
                       path='../input/reef-baseline-fold12/l6_3600_uflip_vm5_f12_up/f1/best.pt',
                       source='local',
                       force_reload=True)  # local repo
model.conf = 0.01

# %%
for idx, (img, pred_df) in enumerate(tqdm(iter_test)):
    anno = ''
    height, width = img.shape[0], img.shape[1]
    r = model(img, size=IMG_SIZE, augment=True)
    if r.pandas().xyxy[0].shape[0] == 0:
        anno = ''
    else:
        for idx, row in r.pandas().xyxy[0].iterrows():
            if row.confidence > 0.15:
                anno += '{} {} {} {} {} '.format(row.confidence, int(row.xmin), int(row.ymin), int(row.xmax-row.xmin), int(row.ymax-row.ymin))
#                 pred.append([row.confidence, row.xmin, row.ymin, row.xmax-row.xmin, row.ymax-row.ymin])
                detects.append([int(row.xmin), int(row.ymin), int(row.xmax), int(row.ymax), row.confidence])
    # Update tracks using detects from current frame
    if len(detects):
        tracked_objects = tracker.update(np.array(detects), np.array([height, width, frame_id]), np.array([IMG_SIZE, IMG_SIZE]))
        for tobj in tracked_objects:
            # Add objects that have no detections on current frame to predictions
            tlwh = tobj.tlwh
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                x_min = int(tlwh[0])
                y_min = int(tlwh[1])
                bbox_width = int(tlwh[2])
                bbox_height = int(tlwh[3])
#                 preds.append([x_min, y_min, bbox_width, bbox_height])
                score = tobj.score
                anno += '{} {} {} {} {} '.format(score, x_min, y_min, bbox_width, bbox_height)
    #         preds.append('{:.2f} {} {} {} {}'.format(score, x_min, y_min, bbox_width, bbox_height))
        #######################################################    
            
    pred_df['annotations'] = anno.strip(' ')
    env.predict(pred_df)
    print('Prediction:', anno.strip(' '))
    frame_id += 1

# %%



