from siamMask.models.custom import Custom
from siamMask.utils.load_helper import load_pretrain
from siamMask.test import siamese_init, siamese_track, get_image_crop
import torch
from polygon import Polygon, Bbox
import cv2
import socketserver
import pickle
import struct
import argparse
import json
import os
from polygon import Bbox


HOST, PORT = "", 8787
OK_SIGNAL = b"ok"
TERMINATE_SIGNAL = "terminate"


class Tracker:
    def __init__(self):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        torch.backends.cudnn.benchmark = True

    def init(self, img, bbox):
        """
        Arguments:
            img (OpenCV image): obtained from cv2.imread(img_file)
            bbox (BBox)
        """
        raise NotImplementedError

    def track(self, img):
        """
        Output:
            bbox (BBox), polygon (Polygon)
        """
        raise NotImplementedError


class SiamMaskTracker(Tracker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cfg = json.load(open("siamMask/configs/config_vot.json"))
        self.tracker = Custom(anchors=self.cfg['anchors'])
        self.tracker = load_pretrain(self.tracker, "siamMask/pretrained/SiamMask_VOT.pth", use_cuda=self.use_cuda)
        self.tracker.eval().to(self.device)

        self.state = None

    def init(self, img, bbox):
        self.state = siamese_init(img, bbox.center, bbox.size, self.tracker, self.cfg['hp'], use_cuda=self.use_cuda)

    def track(self, img):
        self.state = siamese_track(self.state, img, mask_enable=False, refine_enable=False, use_cuda=self.use_cuda, preprocessed=True)
        bbox = Bbox.from_center_size(self.state['target_pos'], self.state['target_sz'])
        polygon = Polygon(self.state['ploygon'].flatten()) if self.state['ploygon'] else Polygon()

        return bbox, polygon


class TrackingHandler(socketserver.BaseRequestHandler):
    PAYLOAD_SIZE = struct.calcsize(">L")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def handle(self):

        tracker = SiamMaskTracker()
        self.send_ok_signal()

        init_bbox = self.receive_bbox()
        image_path = self.receive_image_path()

        if not os.path.exists(image_path):
            self.send_error_signal("No such file on the server {}".format(image_path))
            return

        img = cv2.imread(image_path)
        tracker.init(img, init_bbox)
        self.send_ok_signal()

        while True:
            image_path = self.receive_image_path()

            if image_path == TERMINATE_SIGNAL:
                print("Tracker terminated.")
                return

            if not os.path.exists(image_path):
                self.send_error("No such file on the server {}".format(image_path))
                return

            img = cv2.imread(image_path)
            img = get_image_crop(tracker.state, img)
            bbox, polygon = tracker.track(img)

            self.send_detection(bbox, polygon)

    def receive_bbox(self):
        data = self.request.recv(1024)
        bbox = pickle.loads(data)
        return Bbox(*bbox)

    def send_error(self, error_msg):
        json_data = json.dumps({
            'error': error_msg
        })
        self.request.sendall(json_data.encode())

    def send_detection(self, bbox, polygon):
        json_data = json.dumps({
            'bbox': bbox.to_json(),
            'polygon': polygon.to_json()
        })
        self.request.sendall(json_data.encode())

    def send_error_signal(self, error_msg):
        self.request.sendall(error_msg.encode())

    def send_ok_signal(self):
        self.request.sendall(OK_SIGNAL)

    def receive_image_path(self):
        data = self.request.recv(1024)
        image_path = data.decode()
        return image_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tracking server")
    parser.add_argument("-p", "--port", type=int, default=PORT, help="socket port")
    args = parser.parse_args()

    server = socketserver.TCPServer((HOST, args.port), TrackingHandler)
    server.serve_forever()
