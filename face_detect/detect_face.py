import numpy as np
import cv2
import os
from typing import Tuple


class FaceDetector:
    def __init__(self, prototxt: str, checkpoint_path: str, img_size: Tuple[int, int] = (200, 180),
                 confidence: float = 0.5):
        """
        :param prototxt: path to Caffe 'deploy' prototxt file
        :param checkpoint_path: path to Caffe pre-trained model
        :param confidence: minimum probability to filter weak detections
        """
        assert os.path.exists(prototxt), f'invalid path to prototxt file: {prototxt}'
        assert os.path.exists(checkpoint_path), f'invalid path to checkpoint: {checkpoint_path}'
        self.net = cv2.dnn.readNetFromCaffe(prototxt, checkpoint_path)
        self.img_size = img_size
        self.confidence = confidence

    def detect(self, image_path: str, visualize: bool = False) -> Tuple[int, int, int, int]:
        image = cv2.imread(filename=image_path)
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, self.img_size), 1.0, self.img_size, (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > self.confidence:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the bounding box of the face along with the associated
                # probability
                text = "{:.2f}%".format(confidence * 100)
                if startY - 10 > 10:
                    y = startY - 10
                else:
                    y = startY + 10

                if visualize:
                    cv2.rectangle(image, (startX, startY), (endX, endY),
                                  (0, 0, 255), 2)
                    cv2.putText(image, text, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        if visualize:
            # show the output image
            cv2.imshow("Output", image)
            cv2.waitKey(5000)
        return startX, endX, startY, endY


if __name__ == '__main__':
    face_detector = FaceDetector(prototxt='checkpoints/deploy.prototxt.txt',
                                 checkpoint_path='checkpoints/res10_300x300_ssd_iter_140000.caffemodel')

    ret = face_detector.detect(
        '/Volumes/Samsung_T5/我的文档/大三下/图像处理与模式识别/datasets/FaceRecognitionData/faces96/9540547/9540547.7.jpg',
        visualize=True)
    print(ret)
