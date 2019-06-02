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

    def copy(self):
        return Detection(self.class_id, self.track_id, self.polygon.copy(), self.bbox.copy(), self.keypoints.copy())

    def __repr__(self):
        return "Detection(class_id={}, track_id={}, bbox={}, polygon={}, keypoints={})".format(self.class_id, self.track_id,
                                                                                 self.bbox, self.polygon, self.keypoints)

class TrackInfo:
    def __init__(self, video_name="", file_names=[]):
        self.video_name = video_name
        self.file_names = file_names
        self.nb_frames = len(file_names)

        self.nb_track_ids = 0

        self.class_names = DEFAULT_CLASS_NAMES
        self.detections = [[] for _ in range(self.nb_frames)]

        self.load_from_disk()

    def load_from_disk(self):
        file_name = os.path.join(OUTPUT_DIR, "{}.json".format(self.video_name))

        if not os.path.exists(file_name):
            return

        with open(file_name, "r") as f:
            data = json.load(f)
            self.nb_track_ids = data["nb_track_ids"]
            self._load_class_names(data)
            self.detections = [[Detection.from_json(detection) for detection in frame["detections"]] for frame in data["frames"]]

    def _load_class_names(self, data):
        if "class_names" not in data:
            self.class_names = DEFAULT_CLASS_NAMES

        self.class_names = {int(k): v for k, v in json.loads(data["class_names"]).items()}

    def to_json(self):
        return {
            "video_name": self.video_name,
            "nb_track_ids": self.nb_track_ids,
            "class_names": json.dumps(self.class_names),
            "frames": [
                {
                    "frame_id": i,
                    "file_name": self.file_names[i],
                    "detections": [d.to_json() for d in detections]
                }
                for i, detections in enumerate(self.detections)
            ]
        }

    def get_min_available_track_id(self, frame):
        track_ids = set([d.track_id for d in self.detections[frame]])
        N = len(track_ids)
        missing_track_ids = set(range(N)) - track_ids
        if missing_track_ids:
            return min(missing_track_ids)
        else:
            return N+1

    def save_to_disk(self):
        with open(os.path.join(OUTPUT_DIR, "{}.json".format(self.video_name)), "w") as f:
            json.dump(self.to_json(), f, indent=4)


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

    def detect_single_image(self, image_path, crop_area=None):
        img = Image.open(image_path).convert('RGB')
        w, h = img.size

        # Cropping
        if crop_area is not None:
            print("cropping area {}".format(crop_area))
            img = img.crop(crop_area)

        sized = letterbox_image(img, self.model.width, self.model.height)
        start = time.time()

        with torch.no_grad():
            boxes = do_detect(self.model, sized, 0.5, 0.4, self.use_cuda)

        self.correct_yolo_boxes(boxes, w, h, self.model.width, self.model.height, crop_area)
        finish = time.time()

        print('%s: Predicted in %f seconds.' % (image_path, (finish - start)))

        detections = [Detection(class_id=b[6].item(), track_id=i,
                                bbox=Bbox((b[0] - b[2] / 2).item() * w, (b[1] - b[3] / 2).item() * h,
                                          b[2].item() * w, b[3].item() * h)) for i, b in enumerate(boxes)]

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

            detections = [Detection(track_id=i, class_id=1, keypoints=Keypoints(kp), bbox=utils.keypoints_to_bbox(kp)) for i, kp in enumerate(keypoints)]

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
    # parser.add_argument("-c", "--crop-area", type=, help="crop area")
    args = parser.parse_args()

    if args.detector == "YOLO":
        detector = YOLODetector()
    elif args.detector == "OpenPifPaf":
        detector = OpenPifPafDetector()
    else:
        print("Unknown detector")
        return

    crop_area = None

    file_names = sorted(glob.glob('{}/*.jpg'.format(args.sequence)), key=utils.natural_sort_key)
    video_name = os.path.basename(args.sequence)
    nb_frames = len(file_names)

    track_info = TrackInfo(video_name, file_names)
    start_time = datetime.datetime.now()

    for frame, detections in enumerate(detector.detect_batch(file_names, crop_area)):

        track_info.detections[frame] = detections

        write_running_info(video_name, start_time, frame, nb_frames)

        if frame % 100 == 0:
            track_info.save_to_disk()

    track_info.save_to_disk()


def write_running_info(video_name, start_time, frame, nb_frames):
    data = {
        "video_name": video_name,
        "current_frame": frame + 1,
        "total_frame": nb_frames,
        "start_time": str(start_time),
        "last_update": str(datetime.datetime.now())
    }

    file_name = os.path.join(OUTPUT_DIR, "running_info.json")

    with open(file_name, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()
