import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

def letterbox(img, new_shape=(512, 512), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def detect():
    device = select_device(str(0))
    print('device:', device)
    #half = device.type != 'cpu'
    # Load model
    weights ='final_weight/best.pt'
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(512, s=model.stride.max())  # check img_size
    save_img = True
    
    ##dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img) #if device.type != 'cpu' else None  # run once
    if os.path.exists('result/'):
        shutil.rmtree('result')
    if not os.path.exists('result'):
        os.makedirs('result')
    if os.path.exists('predicted/'):
        shutil.rmtree('predicted')
    if not os.path.exists('predicted'):
        os.makedirs('predicted')
    img_list = [file for file in os.listdir('test_img/') if file.endswith('.jpg')]
    for li in img_list:  
        txt_name=li[:-4]+'.txt'
        
        img0 = cv2.imread('test_img/'+li)
        img = letterbox(img0, new_shape=512)[0]

            # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.4, 0.5, classes=0, agnostic=False)
        t2 = time_synchronized()
        for i, det in enumerate(pred):
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                with open('predicted/'+txt_name,'w') as f:
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        label = '%s %.2f' % (names[int(cls)], conf)
                        x1, y1,x2,y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        f.write('{} {} {} {} {} {}'.format(str(names[int(cls)]),conf,x1,y1,x2,y2))
                        f.write('\n')
                        plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

        cv2.imwrite('result/{}'.format(li), img0)
        print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    with torch.no_grad():
        detect()
