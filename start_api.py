from flask import Flask, jsonify, request
from pathlib import Path
from shutil import rmtree
import uuid
import time
import torch
import torch.backends.cudnn
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords


# Initialize Flask app
app = Flask(__name__)
Path('./temp').mkdir(exist_ok=True)

# Initialize YOLOv7
device = 'cpu' # CPU is more cost-effective for this project
weights = ['yolov7.pt']
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(480, s=stride)  # check img_size
# set_logging()


def detect(source):    
    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    vehicle_counts = {}
    for path, img, im0s, _ in dataset:
        image_name = path.split("/")[-1]
        t0 = time.time()
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

        # Process detections
        vehicle_count = 0
        for _, det in enumerate(pred):  # detections per image
            p, s, im0, _ = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Count results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    if c in [2, 3, 5, 7]:
                        vehicle_count += int(n)
                    s += f'{n} {names[int(c)]}{"s" * (n > 1)}, '  # add to string
        vehicle_counts[image_name] = vehicle_count
        print(f'[{image_name}] {s}Done. ({time.time() - t0:.3f}s)')
    return vehicle_counts


@app.route('/health', methods=['GET'])
def health():
    return "API is up!"


@app.route('/count-vehicles', methods=['POST'])
def count_vehicles():
    if 'images' not in request.files:
        return "Image files not provided", 400
    
    t0 = time.time()

    # Initialize temp directory for image files
    temp_dir = f'./temp/{uuid.uuid4().hex}'
    Path(temp_dir).mkdir(exist_ok=True)
    
    received_files = request.files.getlist("images")
    for f in received_files:
        file_name = f.headers['Content-Disposition'].split("filename=")[1][1:-1]
        file_path = f'{temp_dir}/{file_name}'
        f.save(file_path)

    with torch.no_grad():
        vehicle_counts = detect(temp_dir)
        time_taken = f'{time.time() - t0:.3f}s'
        rmtree(temp_dir)
        return jsonify({'vehicle_counts' : vehicle_counts, 'time_taken': time_taken})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7777, debug=False)