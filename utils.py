import struct
import os
import cv2
import re
import numpy as np
from polygon import Bbox


def send_data(socket, data):
    data = struct.pack('>I', len(data)) + data
    socket.sendall(data)


def recv_data(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)


def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]


def keypoints_to_bbox(kps):
    x, y, v = kps[0::3], kps[1::3], kps[2::3]
    x, y = x[v > 0], y[v > 0]

    x1, y1, x2, y2 = np.min(x), np.min(y), np.max(x), np.max(y)
    return Bbox(x1, y1, x2 - x1, y2 - y1)
