import sys
import math
import time


class BoundingBox():
    def __init__(self, xmin, ymin, xmax, ymax, detectionLabel, detectionProb):
        self.updatePosition(xmin, ymin, xmax, ymax)
        self.detectionLabel = detectionLabel
        self.detectionProb = detectionProb
        self.missingFrames = 0
        self.enteringFrames = 0
        self.id = None
        self.selected = False
        self.valid = False
        self.time = 0

    def incMissingFrames(self):
        self.missingFrames += 1

    def incEnteringFrames(self):
        self.enteringFrames += 1

    def resetMissingFrames(self):
        self.missingFrames = 0

    def resetEnteringFrames(self):
        self.enteringFrames = 0

    def setID(self, id):
        self.id = id

    def setValid(self, valid):
        self.valid = valid

    def setSelected(self, selected):
        self.selected = selected

    def updatePosition(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.center = [self.xmin + (self.xmax - self.xmin) / 2, self.ymin + (self.ymax - self.ymin) / 2]

    def distance(self, bb):
        return math.sqrt(((self.center[0] - bb.center[0]) ** 2 + (self.center[1] - bb.center[1]) ** 2))


class BoundingBoxTracker():
    def __init__(self, rep_duration, min_probability=0.3, min_frames=3, max_missing_frames=10):
        self.nextid = 1
        self.bbs = {}
        self.max_missing_frames = max_missing_frames
        self.min_frames = min_frames
        self.min_probability = min_probability
        self.total = 0
        self.rep_duration = rep_duration

    def updateBBs(self, bounding_boxes: [BoundingBox]):
        # print(str(len(bounding_boxes)))
        for key in self.bbs:
            self.bbs[key].setSelected(False)
        for bb in bounding_boxes:
            # Compare the bounding box center distance with all the recorded bounding boxes and choose the smaller
            last_distance = sys.float_info.max
            for key in self.bbs:
                current_bb = self.bbs[key]
                if current_bb.selected:
                    continue
                distance = bb.distance(current_bb)
                if distance < last_distance:
                    last_distance = distance
                    bb.id = current_bb.id
                    bb.enteringFrames = current_bb.enteringFrames
                    bb.valid = current_bb.valid
                    bb.time = current_bb.time
            # Add a new bounding box with a new ID to the list
            if bb.id is None:
                bb.setID("t_" + str(self.nextid))
                self.nextid += 1
            bb.setSelected(True)
            self.bbs[bb.id] = bb
        keys = list(self.bbs)
        for key in keys:
            current_bb = self.bbs[key]
            # print(str(current_bb.id) + ' ' + str(current_bb.detectionProb) + ' ' + str(current_bb.enteringFrames))
            if current_bb.detectionProb >= self.min_probability:
                current_bb.incEnteringFrames()
                if current_bb.enteringFrames >= self.min_frames:
                    if not current_bb.valid:
                        self.total += 1
                        del self.bbs[current_bb.id]
                        current_bb.setID(self.total)
                        current_bb.setValid(True)
                        current_bb.time = time.time()
                        self.bbs[current_bb.id] = current_bb
            else:
                current_bb.resetEnteringFrames()
            if not current_bb.selected:
                if not current_bb.valid:
                    del self.bbs[key]
                    continue
                current_bb.incMissingFrames()
                if current_bb.missingFrames >= self.max_missing_frames:
                    self.rep_duration(time.time() - current_bb.time)
                    del self.bbs[key]
                else:
                    current_bb.incMissingFrames()

        return self.total

    def getBBs(self):
        filtered = dict()
        for (key, value) in self.bbs.items():
            bb = self.bbs[key]
            if bb.valid:
                filtered[key] = bb
        return filtered

    def getTotalCount(self):
        return self.total

    def getTotalDuration(self):
        duration = 0
        for key in self.bbs:
            bb = self.bbs[key]
            if bb.valid:
                duration += time.time() - bb.time
        return duration
