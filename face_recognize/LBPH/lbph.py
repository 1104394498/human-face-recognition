from skimage import feature
import numpy as np
from sklearn.svm import LinearSVC
import os
import csv
from PIL import Image
from torchvision.transforms import transforms
import pickle
from face_detect.detect_face import FaceDetector
from typing import Tuple
from results.ResultWriter import ResultWriter
from config.Config import Config


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # return the histogram of Local Binary Patterns
        return hist


class LBPHRecognizer:
    def __init__(self, config, C=100.0, random_state=42, max_iter=6000, num_points=24, radius=8, face_detect=False,
                 transform=None, img_size=(200, 180)):
        # type: (Config, float, int, int, int, int, bool, transforms.Compose, Tuple[int, int])->None
        self.clf = LinearSVC(C=C, random_state=random_state, max_iter=max_iter)
        self.lbp = LocalBinaryPatterns(numPoints=num_points, radius=radius)
        self.transform = transform
        self.img_size = img_size
        if face_detect:
            self.face_detector = FaceDetector(
                prototxt='face_detect/checkpoints/deploy.prototxt.txt',
                checkpoint_path='face_detect/checkpoints/res10_300x300_ssd_iter_140000.caffemodel',
                img_size=img_size
            )
        else:
            self.face_detector = None

        self.result_writer = ResultWriter(config=config, method='lbph')

    def _get_data_names_for_csv(self, csv_path):
        assert os.path.exists(csv_path), f'invalid csv path: {csv_path}'
        data, names, image_paths = [], [], []
        with open(csv_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for image_path, name in csv_reader:
                try:
                    img = Image.open(image_path).convert('L')
                    if self.face_detector is not None:
                        x_begin, x_end, y_begin, y_end = self.face_detector.detect(image_path, False)
                        img = img.crop((x_begin, y_begin, x_end, y_end))
                    if self.transform is not None:
                        img = self.transform(img)
                        img = self.transform(img)
                    # img.show()
                    img = np.array(img)
                except:
                    continue
                image_paths.append(image_path)
                hist = self.lbp.describe(img)
                data.append(hist)
                names.append(name)
        return data, names, image_paths

    def train(self, csv_path):
        # type:(str)->None
        data, names, _ = self._get_data_names_for_csv(csv_path)
        self.clf.fit(data, names)

    def predict(self, csv_path, data_type):
        # type: (str, str) -> float
        """
        :return: accuracy
        """
        data, names, image_paths = self._get_data_names_for_csv(csv_path)
        pred_names = self.clf.predict(data)
        right_num = 0
        assert len(names) == len(pred_names)
        for i in range(len(pred_names)):
            if names[i] == pred_names[i]:
                right_num += 1
        acc = right_num / len(pred_names)
        self.result_writer.write(image_paths, pred_names, names, acc, data_type + '.csv')
        return acc

    def save_model(self, file_name: str):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        face_detector = self.face_detector
        self.face_detector = None
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)
        self.face_detector = face_detector

    def load_model(self, file_name: str):
        assert os.path.exists(file_name), f'invalid file name: {file_name}'
        with open(file_name, 'rb') as f:
            checkpoint = pickle.load(f)
        self.clf = checkpoint.clf
        self.lbp = checkpoint.lbp
        self.transform = checkpoint.transform
