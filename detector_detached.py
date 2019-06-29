import torch
from polygon import Polygon, Bbox
import json
import socket
import pickle
import cv2
import struct
import socketserver
import datetime
import argparse
from polygon import Bbox, Keypoints
import utils
from PIL import Image
import os
import time
from detection.yolov3.batch_detect import model
from detection.yolov3.utils import do_detect
from detection.yolov3.image import letterbox_image
import numpy as np
import openpifpaf.network
import openpifpaf
import glob
from config import OUTPUT_DIR
from class_names import DEFAULT_CLASS_NAMES
import math
import pandas as pd


class Detection:
    def __init__(self, class_id=0, track_id=0, polygon=Polygon(), bbox=Bbox(), keypoints=Keypoints()):
        self.class_id = class_id
        self.track_id = track_id
        self.polygon = polygon
        self.bbox = bbox
        self.keypoints = keypoints

    @staticmethod
    def from_json(data):
        return Detection(data["class_id"], data["track_id"],
                         Polygon(data["polygon"]), Bbox(*data["bbox"]), Keypoints(data["keypoints"]))

    def to_json(self):
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "polygon": self.polygon.to_json(),
            "bbox": self.bbox.to_json(),
            "keypoints": self.keypoints.to_json()
        }

    def to_dict(self):
        return {"track_id": self.track_id, "class_id": self.class_id, **self.bbox.to_dict(),
                "polygon": self.polygon.to_str(), "kp": self.keypoints.to_str()}

    @staticmethod
    def from_df(row):
        bbox = Bbox(row.x, row.y, row.w, row.h)
        return Detection(row.class_id, row.track_id, Polygon.from_str(row.polygon), bbox, Keypoints.from_str(row.kp))

    def copy(self):
        return Detection(self.class_id, self.track_id, self.polygon.copy(), self.bbox.copy(), self.keypoints.copy())

    def __repr__(self):
        return "Detection(class_id={}, track_id={}, bbox={}, polygon={}, keypoints={})".format(self.class_id, self.track_id,
                                                                                 self.bbox, self.polygon, self.keypoints)


class TrackInfo:
    def __init__(self, video_name=""):
        self.video_name = video_name

        dir_name = os.path.join(OUTPUT_DIR, self.video_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        self.nb_track_ids = 0
        self.class_names = DEFAULT_CLASS_NAMES
        self.load_info()

        self.file_name = None
        self.detections = []

    def save_to_disk(self):
        self.write_info()
        self.write_detections(self.file_name)

    def load_info(self):
        json_file = os.path.join(OUTPUT_DIR, "{}/info.json".format(self.video_name))

        if not os.path.exists(json_file):
            return

        with open(json_file, "r") as f:
            data = json.load(f)
            self.nb_track_ids = data["nb_track_ids"]
            self.class_names = {int(k): v for k, v in json.loads(data["class_names"]).items()}

    @staticmethod
    def df_from_csv(file_name):
        if not os.path.exists(file_name):
            return pd.DataFrame(columns=["track_id", "class_id", "x", "y", "w", "h", "polygon", "kp"])

        return pd.read_csv(file_name, header=None, names=["track_id", "class_id", "x", "y", "w", "h", "polygon", "kp"],
                           na_filter=False)

    @staticmethod
    def df_to_csv(df, file_name):
        df.to_csv(file_name, index=None, header=False)

    @staticmethod
    def df_add_detection(df, detection: Detection):
        return df.append(detection.to_dict(), ignore_index=True)

    def get_detections(self, file_name):
        txt_file = os.path.join(OUTPUT_DIR, "{}/{}.txt".format(self.video_name, file_name))

        if not os.path.exists(txt_file):
            return []

        df = self.df_from_csv(txt_file)
        return [Detection.from_df(row) for _, row in df.iterrows()]

    def load_detections(self, file_name):
        self.file_name = file_name
        self.detections = self.get_detections(file_name)

    def write_info(self):
        json_file = os.path.join(OUTPUT_DIR, "{}/info.json".format(self.video_name))

        data = {
            "video_name": self.video_name,
            "nb_track_ids": self.nb_track_ids,
            "class_names": json.dumps(self.class_names)
        }

        with open(json_file, "w") as f:
            json.dump(data, f)

    def write_detections(self, file_name, detections=None):
        txt_file = os.path.join(OUTPUT_DIR, "{}/{}.txt".format(self.video_name, file_name))

        if detections is None:
            detections = self.detections

        if file_name == self.file_name:
            self.detections = detections

        df = pd.DataFrame(columns=["track_id", "class_id", "x", "y", "w", "h", "polygon", "kp"])
        if len(detections) > 0:
            df = df.append([d.to_dict() for d in detections], ignore_index=True)
        self.df_to_csv(df, txt_file)

        self.nb_track_ids = max(self.nb_track_ids, max([d.track_id for d in detections] or [0]) + 1)


class Detector:
    def __init__(self):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        torch.backends.cudnn.benchmark = True

    def detect_single_image(self, image_path, crop_area=None):
        """
        Output:
            detections ([Detection])
        """
        raise NotImplementedError

    def detect_batch(self, images, crop_area=None):
        for image_path in images:
            yield self.detect_single_image(image_path, crop_area)

    def terminate(self):
        pass


class YOLODetector(Detector):
    CONFIG_FILE = "detection/yolov3/yolo_cfg/topview-6-predict.cfg"
    WEIGHTS_FILE = "detection/yolov3/weights/topview-final-6.weights"

    def __init__(self):
        super().__init__()

        self.model = model(self.CONFIG_FILE, self.WEIGHTS_FILE)
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.model.cuda()

    def correct_yolo_boxes(self, boxes, im_w, im_h, net_w, net_h, cropping_area=None):
        im_w, im_h = float(im_w), float(im_h)
        net_w, net_h = float(net_w), float(net_h)

        if cropping_area is None:
            area_w, area_h = im_w, im_h
            factor_w, factor_h = 1, 1
            offset_w, offset_h = 0, 0
        else:
            area_w, area_h = cropping_area[2] - cropping_area[0], cropping_area[3] - cropping_area[1]
            factor_w, factor_h = area_w / im_w, area_h / im_h
            offset_w, offset_h = cropping_area[0] / im_w, cropping_area[1] / im_h

        if net_w/area_w < net_h/area_h:
            new_w = net_w
            new_h = (area_h * net_w)/area_w
        else:
            new_w = (area_w * net_h)/area_h
            new_h = net_h

        xo, xs = (net_w - new_w)/(2*net_w), net_w/new_w
        yo, ys = (net_h - new_h)/(2*net_h), net_h/new_h

        for i in range(len(boxes)):
            b = boxes[i]
            b[0] = (b[0] - xo) * xs * factor_w + offset_w
            b[1] = (b[1] - yo) * ys * factor_h + offset_h
            b[2] *= xs * factor_w
            b[3] *= ys * factor_h

    def yolo_to_coco_class(self, class_id):
        return class_id

    def detect_single_image(self, image_path, crop_area=None):
        img = Image.open(image_path).convert('RGB')
        W, H = img.size

        # Cropping
        if crop_area is None:
            crop_area = (0, 0, W, H)
        x_crop, y_crop, w_crop, h_crop = crop_area

        margin = 0.03 * max(W, H)  # 3% margin

        detections = []

        # Number of repeated cropping areas to span the entire image
        n_left = math.ceil(x_crop / w_crop)
        n_right = math.ceil((W - (x_crop + w_crop)) / w_crop)
        n_top = math.ceil(y_crop / h_crop)
        n_bottom = math.ceil((H - (y_crop + h_crop)) / h_crop)

        for i in range(-n_top, 1 + n_bottom):
            for j in range(-n_left, 1 + n_right):
                crop_area = (x_crop + j * w_crop - margin, y_crop + i * h_crop - margin,
                             x_crop + j * w_crop + w_crop + margin, y_crop + i * h_crop + h_crop + margin)
                print(i, j, crop_area)

                img_crop = img.crop(crop_area)

                sized = letterbox_image(img_crop, self.model.width, self.model.height)
                start = time.time()

                with torch.no_grad():
                    boxes = do_detect(self.model, sized, 0.5, 0.4, self.use_cuda)

                self.correct_yolo_boxes(boxes, W, H, self.model.width, self.model.height, crop_area)
                finish = time.time()

                print('%s: Predicted in %f seconds.' % (image_path, (finish - start)))

                detections.extend([Detection(class_id=self.yolo_to_coco_class(b[6].item()), track_id=i,
                                             bbox=Bbox((b[0] - b[2] / 2).item() * W, (b[1] - b[3] / 2).item() * H,
                                                       b[2].item() * W, b[3].item() * H)) for i, b in enumerate(boxes)])

        return detections


class OpenPifPafDetector(Detector):

    def __init__(self):
        super().__init__()

        parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        openpifpaf.network.nets.cli(parser)
        openpifpaf.decoder.cli(parser, force_complete_pose=False, instance_threshold=0.2)
        parser.add_argument('--batch-size', default=1, type=int,
                            help='processing batch size')
        parser.add_argument('--loader-workers', default=2, type=int,
                            help='number of workers for data loading')
        self.args = parser.parse_args([])

        # add args.device
        self.args.device = torch.device('cpu')
        self.args.pin_memory = False
        if torch.cuda.is_available():
            self.args.device = torch.device('cuda')
            self.args.pin_memory = True

        model, _ = openpifpaf.network.nets.factory_from_args(self.args)
        model = model.to(self.args.device)
        self.processor = openpifpaf.decoder.factory_from_args(self.args, model)

    def detect(self, image_paths, image_tensors, processed_images_cpu):
        with torch.no_grad():
            images = image_tensors.permute(0, 2, 3, 1)

            processed_images = processed_images_cpu.to(self.args.device, non_blocking=True)
            fields_batch = self.processor.fields(processed_images)
            pred_batch = self.processor.annotations_batch(fields_batch, debug_images=processed_images_cpu)

            _, image, processed_image_cpu, pred = image_paths[0], images[0], processed_images_cpu[0], pred_batch[0]

            self.processor.set_cpu_image(image, processed_image_cpu)
            keypoint_sets, scores = self.processor.keypoint_sets_from_annotations(pred)  # keypoints shape: (nb_detections, 17, 3)

            keypoints = keypoint_sets.reshape(-1, 3 * 17)  # keypoints shape: (nb_detections, 3*17)
            print(keypoints)

            detections = [Detection(track_id=i, class_id=5, keypoints=Keypoints(kp), bbox=utils.keypoints_to_bbox(kp)) for i, kp in enumerate(keypoints)]

            return detections

    def detect_single_image(self, image_path, crop_area=None):
        data = openpifpaf.datasets.ImageList([image_path])
        data_loader = torch.utils.data.DataLoader(
            data, batch_size=self.args.batch_size, shuffle=False,
            pin_memory=self.args.pin_memory, num_workers=self.args.loader_workers)

        image_paths, image_tensors, processed_images_cpu = next(iter(data_loader))
        return self.detect(image_paths, image_tensors, processed_images_cpu)

    def detect_batch(self, images, crop_area=None):
        data = openpifpaf.datasets.ImageList(images)
        data_loader = torch.utils.data.DataLoader(
            data, batch_size=self.args.batch_size, shuffle=False,
            pin_memory=self.args.pin_memory, num_workers=self.args.loader_workers)

        for image_paths, image_tensors, processed_images_cpu in data_loader:
            yield self.detect(image_paths, image_tensors, processed_images_cpu)


def main():
    parser = argparse.ArgumentParser(description="Detection server")
    parser.add_argument("-s", "--sequence", type=str, help="path to the sequence")
    parser.add_argument("-d", "--detector", type=str, help="type of detector to use")
    parser.add_argument("-c", "--crop-area", type=int, nargs=4, help="crop area")
    args = parser.parse_args()

    print(args)

    if args.detector == "YOLO":
        detector = YOLODetector()
    elif args.detector == "OpenPifPaf":
        detector = OpenPifPafDetector()
    else:
        print("Unknown detector")
        return

    crop_area = args.crop_area

    if not os.path.exists(args.sequence):
        write_running_info(error="No such file or directory: {}".format(args.sequence))
        return

    file_names = sorted(glob.glob('{}/*.jpg'.format(args.sequence)), key=utils.natural_sort_key)
    video_name = os.path.basename(args.sequence)
    nb_frames = len(file_names)

    track_info = TrackInfo(video_name)
    start_time = datetime.datetime.now()

    for frame, detections in enumerate(detector.detect_batch(file_names, crop_area)):
        file_path = file_names[frame]
        base = os.path.basename(file_path)
        file_name = os.path.splitext(base)[0]

        track_info.write_detections(file_name, detections)
        write_running_info(video_name, start_time, frame, nb_frames)

    track_info.save_to_disk()


def write_running_info(video_name="", start_time="", frame=0, nb_frames=0, error=None):
    data = {
        "video_name": video_name,
        "current_frame": frame + 1,
        "total_frame": nb_frames,
        "start_time": str(start_time),
        "last_update": str(datetime.datetime.now()),
    }

    if error is not None:
        data["error"] = error

    file_name = os.path.join(OUTPUT_DIR, "running_info.json")

    with open(file_name, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        write_running_info(error=str(e))
