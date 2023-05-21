# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import time
from collections import deque




FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode, time_sync




#############################################




class EuclideanDistTracker:
    def __init__(self):
        self.center_points={}
        self.id_count=0
    def update(self,obj_rect):
        obj_bbx_ids=[]
        for rect in obj_rect:
            x,y,w,h=rect
            cx=(x+x+w)//2
            cy=(y+y+h)//2
            same_obj=False
            for id,pt in self.center_points.items():
                dist=math.hypot(cx-pt[0],cy-pt[1])
                # if dist<200:
                # if dist < 120:
                if dist < 25:
                    self.center_points[id]=(cx,cy)
                    obj_bbx_ids.append([x,y,w,h,id])
                    same_obj=True
                    break
            if same_obj is False:
                self.center_points[self.id_count]=(cx,cy)
                obj_bbx_ids.append([x,y,w,h,self.id_count])
                self.id_count+=1
        new_cnt_point={}
        for bb_id in obj_bbx_ids:
            _,_,_,_,obj_id=bb_id
            center=self.center_points[obj_id]
            new_cnt_point[obj_id]=center
        self.center_points=new_cnt_point.copy()

        return obj_bbx_ids

class SpeedEstimator:
    def __init__(self,posList,fps):
        self.x=posList[0]
        self.y=posList[1]
        self.fps=fps
        
    def estimateSpeed(self):
        # Distance / Time -> Speed
        d_pixels=math.sqrt(self.x+self.y)
        ppm=1.8
        # ppm=4.8
        d_meters=int(d_pixels*ppm)
        speed=d_meters/self.fps*3.6
        speedInKM=np.average(speed)
        return int(speedInKM)




##########################################################








@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    # save_crop = True # my addition
    view_img= True # my addition
    # save_img= True # my addition
    tracker=EuclideanDistTracker()  
    PTime=3


    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]

    g = 0

    # pts = []

    # print(pts)



    pts = [deque(maxlen=30) for _ in range(100000)]

    prev = [0]*100000
    post = [0]*100000
    

    def get_value(main_cor1, main_cor2, cor):
        kk = ((cor[0]-main_cor2[0])*((main_cor1[1]-main_cor2[1])/(main_cor1[0]-main_cor2[0])))-(cor[1]-main_cor2[1])
        print(kk)
        return kk



    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        
        # cv2.imshow("image", im)
        # print(im)
        
        
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        
        
        # cv2.imshow("image", np.asarray(im)) ######################
        # print(im)
        # plt.imshow(im)


        pred = model(im, augment=augment, visualize=visualize) # preditions akibbbbbbb
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions

        








        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            # print("name is : ==> ", p.name) #=============================================

            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            

            # stream images .............................................................

            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)
            
            # stream images .............................................................

            # crossingLine = [(261,474), (2558, 474)]
            # crossingLine = [(647,1411), (1332,24)]
            # crossingLine = [(1489,1405), (1512,32)]
            # crossingLine = [[(494, 996), (1321, 52)],[(1489,1405), (1512,32)],[(1669, 99), (2229, 1369)]]
            # crossingLine = [(1669, 99), (2229, 1369)]



            crossingLine = [(543, 9), (655, 710)]


            
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            # cv2.imshow("ppp",annotator)
            # print(annotator.plot_labels)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # xyxy = det[:, 0:4]
                # conf = det[:, 4]
                # cls = det[:, 5]

                # print(xyxy)
                # print(conf) # gives xy, conf, cls
                # print(cls)


                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
       
                    
                det2 = []
                for *xyxy, conf, cls in det:
                    # print(xyxy)
                    x, y, w, h = xyxy
                    x = int(x)
                    y = int(y)
                    w = int(w)
                    h = int(h)
                    det2.append([x,y,w,h])
                
                CTime=time.time()
                fps=1/(CTime-PTime)
                PTime=CTime

                boxes_ids=tracker.update(det2)

                det3 = []

                for kk in range(len(det2)):
                    det3.append([boxes_ids[kk],det[kk],det[kk][1],det[kk][2]])

                # for kk in range(len(det)):
                    # print(det[kk])
                k = len(det)-1
                cv2.line(im0, crossingLine[0], crossingLine[1], (0, 190, 255), 10)

                

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # print(xyxy) ## print pixel values of single frame images
                    # print(det3[k][0])
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh ##### xyxy ke xywh e conver kore
                    x, y, w, h , id = det3[k][0]
                    k = k-1
                    x = int(x)
                    y = int(y)
                    w = int(w)
                    h = int(h)
                    # print("ex1")

                    # xyxy = [x,y,w,h]        

                    if(id < 10000):
                        center = (int((x+w)/2),int((y+h)/2)) 
                    
                        pts[id].appendleft(center)
                        # print("lenght : {}, pts[{}] : {}".format(len(pts[id]),id,pts[id]))
                        # if(len(pts[id]) > 2):
                        #     print("first : {}, second : {}".format(pts[id][0], pts[id][1]))
                        # m = 0
                        # if(len(pts[id]) > 1):
                        for gg in range(1, len(pts[id])):
                            # m = m+1
                            # print("i : {}".format(i))
                    #         # if either of the tracked points are None, ignore
                    #         # them
                    #         # print(pts[id][i])
                            if pts[id][gg - 1] is not None and pts[id][gg] is not None:                
                            #     # otherwise, compute the thickness of the line and
                            #     # draw the connecting lines
                                pos_value = get_value(crossingLine[0], crossingLine[1],pts[id][gg-1])
                                pos_value = int(pos_value)
                                if(pos_value > 0):
                                    prev[id] = 1
                                    print("pos_value > 0 : {} > {} with id: {}".format(pos_value, 0, id))
                                if(pos_value <= 0):
                                    post[id] = 1
                                    print("pos_value <= 0 : {} <= {} with id: {} and center-cor: {}".format(pos_value, 0, id, pts[id][gg-1]))

                                
                                thickness = int(np.sqrt(40 / float(i + 1)) * 2.5)
                                cv2.line(im0, pts[id][gg - 1], pts[id][gg], (0, 0, 255), thickness)
                    # print("ex2")

                    cv2.waitKey(1)

                    # violation : lane jump portion
                    # if(y < 440 and 440 < y+h):
                        # annotator.box_label(xyxy, names[int(cls)], color=(255,0,0))
                    


                    # violation : speed estimation with unauthorized vehicle classes 

                    SpeedEstimatorTool=SpeedEstimator([x,y],fps)
                    speed=SpeedEstimatorTool.estimateSpeed()

                    vehicle_speed = names[int(cls)]+":"+str(speed)+"Km/h,id:"+str(id)
                    print("XYXY : ",xyxy)
                    # if(1 and names[int(cls)] == "car" ):
                    if(post[id] and prev[id]):
                        annotator.box_label(xyxy, vehicle_speed, color=(0,0,255))
                    else:
                        annotator.box_label(xyxy, vehicle_speed, color=(0,255,0))

                        # if i want to save the violated vehicle images then this is the line
                        # save_one_box(xyxy, imc, file=save_dir / 'crops' / names[int(cls)] / f'{p.stem}.jpg', BGR=True)
                    
                    # cv2.imshow("nothing", im0)
                    # cv2.waitKey(1000)
                    # print("ex3")
                    
            # print("ex4")

            # Stream results
            im0 = annotator.result()
            # print("Result in after annotator : ", im0)
            
            
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
            print("ex5")

                
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path :  # new video
                        vid_path[i] = save_path
                        # vid_path[i] = "/home/akib/Documents"

                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
            print("ex6")
            

        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
