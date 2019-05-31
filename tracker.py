from siamMask.models.custom import Custom
from siamMask.utils.load_helper import load_pretrain
from siamMask.test import siamese_init, siamese_track
import torch
from polygon import Polygon, Bbox
import cv2
import socketserver
import pickle
import struct
import argparse
import json
from polygon import Bbox


HOST, PORT = "", 8787
OK_SIGNAL = b"ok"
TERMINATE_SIGNAL = b"terminate"


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
        img = self.receive_frame()
        tracker.init(img, init_bbox)
        self.send_ok_signal()

        while True:
            img = self.receive_frame()

            if img is None:
                print("Tracker terminated.")
                break

            bbox, polygon = tracker.track(img)

            self.send_detection(bbox, polygon)

    def receive_bbox(self):
        data = self.request.recv(1024)
        bbox = pickle.loads(data)
        return Bbox(*bbox)

    def send_detection(self, bbox, polygon):
        json_data = json.dumps({
            'bbox': bbox.to_json(),
            'polygon': polygon.to_json()
        })
        self.request.sendall(json_data.encode())

    def send_ok_signal(self):
        self.request.sendall(OK_SIGNAL)

    def receive_frame(self):
        data = b""

        while len(data) < self.PAYLOAD_SIZE:
            data += self.request.recv(4096)

        if data == TERMINATE_SIGNAL:
            return

        packed_msg_size = data[:self.PAYLOAD_SIZE]
        msg_size = struct.unpack(">L", packed_msg_size)[0]

        data = data[self.PAYLOAD_SIZE:]
        while len(data) < msg_size:
            data += self.request.recv(4096)
        frame_data = data[:msg_size]

        frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tracking server")
    args = parser.parse_args()

    server = socketserver.TCPServer((HOST, PORT), TrackingHandler)
    server.serve_forever()
