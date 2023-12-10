from flask import Flask, jsonify, request
from pathlib import Path
from shutil import rmtree

import argparse
import time
import torch
import torch.backends.cudnn

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import time_synchronized


# Initialize Flask app and argument object
app = Flask(__name__)
opt = None

# Initialize YOLOv7
device = 'cpu' # CPU is more cost-effective for this project
weights = ['yolov7.pt']
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(480, s=stride)  # check img_size
# set_logging()


def detect(source):
    t0 = time.time()
    
    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    for path, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for _, det in enumerate(pred):  # detections per image
            p, s, im0, _ = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                vehicle_count = 0
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    if c in [2, 3, 5, 7]:
                        vehicle_count += int(n)
                    s += f'{n} {names[int(c)]}{"s" * (n > 1)}, '  # add to string

    print(f'[{source}] {s}Done. ({time.time() - t0:.3f}s)')
    return vehicle_count


@app.route('/health', methods=['GET'])
def health():
    return "API is up!"


@app.route('/count-vehicles', methods=['POST'])
def count_vehicles():
    if 'images' not in request.files:
        return "Image files not provided", 400
    
    t0 = time.time()

    # Initialize temp directory for image files
    Path("./temp").mkdir(exist_ok=True)
    vehicle_counts = {}
    
    received_files = request.files.getlist("images")
    for f in received_files:
        file_name = f.headers['Content-Disposition'].split("filename=")[1][1:-1]
        file_path = f'./temp/{file_name}'
        f.save(file_path)
        with torch.no_grad():
            vehicle_counts[file_name.split('.')[0]] = detect(file_path)
    time_taken = f'{time.time() - t0:.3f}s'
    return jsonify({'vehicle_counts' : vehicle_counts, 'time_taken': time_taken})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=480, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    app.run(host='0.0.0.0', port=7777, debug=False)