import numpy as np
import time
from openvino.inference_engine import IECore
import os
import cv2
import argparse
import traceback


class Queue:
    '''
    Class for dealing with queues
    '''

    def __init__(self):
        self.queues = []

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max = q
            frame = image[y_min:y_max, x_min:x_max]
            yield frame

    def check_coords(self, coords):
        d = {k + 1: 0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0] > q[0] and coord[2] < q[2]:
                    d[i + 1] += 1
        return d


class PersonDetect:
    '''
    Class for the Person Detection Model.
    '''

    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.threshold = threshold
        self.input_name = None
        self.input_shape = None
        self.output_name = None
        self.output_shape = None
        self.exec_network = None
        self.model_width = None
        self.model_height = None
        self.initial_w = None
        self.initial_h = None

    def load_model(self):
        # Initialize the inference engine
        ie = IECore()

        # Read the IR as a IENetwork
        self.exec_network = ie.read_network(model=self.model_structure, weights=self.model_weights)

        ### Check for any unsupported layers, and let the user
        ### know if anything is missing. Exit the program, if so.
        supported_layers = ie.query_network(network=self.exec_network, device_name=self.device)
        unsupported_layers = [l for l in self.exec_network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)

        # Load the IENetwork into the inference engine
        self.exec_network = ie.load_network(self.exec_network, self.device)

        # Get the layer's info
        self.input_name = next(iter(self.exec_network.inputs))
        self.output_name = next(iter(self.exec_network.outputs))
        self.input_shape = self.exec_network.inputs[self.input_name].shape
        self.output_shape = self.exec_network.outputs[self.output_name].shape

        print(f"Input shape: {self.input_shape}")
        print(f"Output shape: {self.output_shape}")
        self.model_width = self.input_shape[3]
        self.model_height = self.input_shape[2]
        print(f'Input image will be resized to ( {self.model_width} x {self.model_height} ) for inference')
        return

    def predict(self, image):
        outputs = []
        frame_inference = self.preprocess_input(image)

        # Start asynchronous inference for specified request
        self.exec_network.start_async(request_id=0, inputs={self.input_name: frame_inference})
        if self.exec_network.requests[0].wait(-1) == 0:
            outputs = self.preprocess_outputs(self.exec_network.requests[0].outputs[self.output_name])
            print('ok')
        return outputs

    def draw_outputs(self, boxes, image):
        height, width, depth = image.shape
        for box in boxes:  # Output shape is 1x1x200x7
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        return

    def preprocess_outputs(self, coords):
        valid_boxes = []
        for box in coords[0][0]:  # Output shape is 1x1x200x7
            label_id = int(box[1])
            conf = box[2]
            # Filter per persons detections with specified threshold
            if label_id == 1 and conf >= self.threshold:
                valid_boxes.append(box)
        return valid_boxes

    def preprocess_input(self, image):
        frame_inference = cv2.resize(image, (self.model_width, self.model_height))

        # Transform the image from the original size to the (1, 3, 320, 544) input shape
        frame_inference = frame_inference.transpose((2, 0, 1))
        frame_inference = frame_inference.reshape(1, *frame_inference.shape)
        return frame_inference


def main(args):
    model = args.model
    device = args.device
    video_file = args.video
    max_people = args.max_people
    threshold = args.threshold
    output_path = args.output_path

    start_model_load_time = time.time()
    pd = PersonDetect(model, device, threshold)
    pd.load_model()
    total_model_load_time = time.time() - start_model_load_time

    print(f"Time taken to load model on {total_model_load_time} seconds")
    # exit(0)

    queue = Queue()

    try:
        queue_param = np.load(args.queue_param)
        for q in queue_param:
            queue.add_queue(q)
    except:
        print("error loading queue param file")

    try:
        cap = cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: " + video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)

    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(video_len)
    cap.open(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps,
                                (initial_w, initial_h), True)

    counter = 0
    start_inference_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            counter += 1
            key_pressed = None

            # Do inference and validate boxes
            boxes = pd.predict(frame)

            # Draw boxes
            pd.draw_outputs(boxes, frame)

            num_people = queue.check_coords(boxes)
            print(f"Total People in frame = {len(boxes)}")
            print(f"Number of people in queue = {num_people}")
            out_text = ""
            y_pixel = 25

            for k, v in num_people.items():
                out_text += f"No. of People in Queue {k} is {v} "
                if v >= int(max_people):
                    out_text += f" Queue full; Please move to next Queue "
                cv2.putText(frame, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                out_text = ""
                y_pixel += 40
            out_video.write(frame)

            ## Local mode with window for debug
            if args.preview:
                key_pressed = cv2.waitKey(60)
                cv2.namedWindow('preview')
                cv2.imshow('preview', frame)

            if key_pressed == 27:
                break

        total_time = time.time() - start_inference_time
        total_inference_time = round(total_time, 1)
        fps = counter / total_inference_time

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time) + '\n')
            f.write(str(fps) + '\n')
            f.write(str(total_model_load_time) + '\n')

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='./results')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)
    parser.add_argument("--preview", action='store_true', default=False)

    args = parser.parse_args()

    main(args)
