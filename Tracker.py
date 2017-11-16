import sys, os, cv2
import numpy as np
import namesgenerator
import binascii
from time import time


def get_corners(center, size):
    xmin = center[0] - size[1]/2
    ymin = center[1] - size[0]/2
    xmax = center[0] + size[1]/2
    ymax = center[1] + size[0]/2
    return xmin, ymin, xmax, ymax

def get_center_size(xmin, ymin, xmax, ymax):
    center = [(xmin + xmax)/2, (ymin + ymax)/2]
    size = [ymax - ymin, xmax-xmin]
    return center, size


class TrackedFrame(object):
    def __init__(self, center, img):
        self.img = img
        self.center = np.array(center)
        self.size = np.array(img.shape[:2]).astype(np.float64)
        self.centerSpeed = np.zeros(2)
        self.sizeSpeed = np.zeros(2)
        self.timeNotVisible = 0
        self.timeSinceCreation = 0
        self.hash = binascii.hexlify(os.urandom(16))
        self.name = namesgenerator.get_random_name()

    def update_speed(self, center, img):
        factorCenter = 0.1
        factorSize = 0.03
        centerChange = center - self.center
        sizeChange = np.array(img.shape[:2]) - self.size
        self.centerSpeed = factorCenter * centerChange
        self.sizeSpeed = factorSize * sizeChange
        self.timeNotVisible = 0

    def update(self, timeSpeed):
        self.center += self.centerSpeed
        if self.timeNotVisible == 0: self.size += self.sizeSpeed
        self.timeNotVisible += timeSpeed
        self.timeSinceCreation += 1

    def overlap(self, otherCenter, otherImg, scale='mean'):
        axmin, aymin, axmax, aymax = get_corners(otherCenter, otherImg.shape[:2])
        bxmin, bymin, bxmax, bymax = get_corners(self.center, self.size)
        dx = min(axmax, bxmax) - max(axmin, bxmin)
        dy = min(aymax, bymax) - max(aymin, bymin)
        aSize = otherImg.shape[0] * otherImg.shape[1]
        bSize = self.size[0] * self.size[1]
        meanSize = (aSize + bSize) / 2.
        if scale == 'mean': refSize = meanSize
        elif scale == 'self': refSize = bSize
        if dx >= 0  and dy >= 0:
            return dx * dy / refSize
        else:
            return 0


class Tracker(object):
    def __init__(self):
        self.frames = []

    def new_object(self, center, img):
        center = np.array(center)
        added = False
        bestOverlap = 0
        for frame in self.frames:
            overlap = frame.overlap(center, img)
            if overlap > bestOverlap:
                bestOverlap = overlap
                bestFrame = frame
        if bestOverlap > 0.1:
            bestFrame.update_speed(center, img)
        else:
            self.frames.append(TrackedFrame(center, img))

    def draw_frames(self, src):
        for i,frame in enumerate(self.frames):
            timeSpeed = 1
            otherFrames = [f for f in self.frames if f.hash is not frame.hash]
            for otherFrame in otherFrames:
                if frame.overlap(otherFrame.center, otherFrame.img, scale='self') > 0.5:
                    timeSpeed = 0.1
            frame.update(timeSpeed)
            if frame.timeSinceCreation > 500000./(frame.size[0]*frame.size[1]):
                xmin, ymin, xmax, ymax = get_corners(frame.center, frame.size)
                p1 = (int(xmin), int(ymin))
                p2 = (int(xmax), int(ymax))
                cv2.rectangle(src, p1, p2, (0,0,255), 5)
                pfont = (p1[0], p1[1]-20)
                cv2.putText(src, frame.name, pfont, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        self.frames = [frame for frame in self.frames if frame.timeNotVisible < 30]

