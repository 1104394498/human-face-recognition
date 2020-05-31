import cv2
import numpy as np
import os
from PIL import Image
from typing import List
from tqdm import tqdm
import csv
from torchvision.transforms import transforms
import pickle
from face_detect.detect_face import FaceDetector
from typing import Tuple
from config.Config import Config
from results.ResultWriter import ResultWriter


class EigenFaceRecognizer:
    def __init__(self, config, transform=None, distance_type='norm2', face_detect=True, img_size=(200, 180)):
        # type: (Config, transforms.Compose, str, bool, Tuple[int, int]) -> None
        assert distance_type == 'norm2' or distance_type == 'mahalanobis', f'invalid distance type: {distance_type}'
        self.distance_type = distance_type
        self.avg_face = None
        self.eigen_faces = None
        self.transform = transform
        self.classes_representation = []

        if face_detect:
            self.face_detector = FaceDetector(
                prototxt='face_detect/checkpoints/deploy.prototxt.txt',
                checkpoint_path='face_detect/checkpoints/res10_300x300_ssd_iter_140000.caffemodel',
                img_size=img_size
            )
        else:
            self.face_detector = None

        self.result_writer = ResultWriter(config, 'eigen')

    def get_representation(self, img_path: str):
        if self.avg_face is None or self.eigen_faces is None:
            raise ValueError('Model has not been trained')
        assert os.path.exists(img_path), f'invalid image path: {img_path}'
        image = Image.open(img_path)
        image = image.convert('RGB')
        if self.face_detector is not None:
            x_begin, x_end, y_begin, y_end = self.face_detector.detect(img_path)
            image = image.crop((x_begin, y_begin, x_end, y_end))
        if self.transform is not None:
            image = self.transform(image)
        image = np.array(image, dtype=np.float32)

        image_vector = image.flatten().reshape((-1, 1))
        image_vector -= self.avg_face.reshape((-1, 1))

        return np.dot(self.eigen_faces, image_vector)

    def train(self, train_csv_path: str):
        with open(train_csv_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            image_paths, names = [], []
            for img_path, name in csv_reader:
                image_paths.append(img_path)
                names.append(name)

            eigen_faces, self.avg_face = self.get_eigen_face(image_paths, transform=self.transform)
            self.avg_face = self.avg_face.flatten()
            self.eigen_faces = np.zeros((len(eigen_faces), len(eigen_faces[0])))
            for i in range(len(eigen_faces)):
                self.eigen_faces[i, :] = eigen_faces[i].T
        with open(train_csv_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            cur_rep, cur_num, cur_name = None, 0, None
            for img_path, name in tqdm(csv_reader):
                try:
                    rep = self.get_representation(img_path)
                except:
                    continue
                if cur_name is None:
                    cur_name = name
                elif cur_name != name:
                    self.classes_representation.append((cur_name, cur_rep / cur_num))
                    cur_name = name
                    cur_num = 0
                    cur_rep = None

                if cur_rep is None:
                    cur_rep = rep
                else:
                    cur_rep += rep
                cur_num += 1

    def predict(self, csv_path: str, data_type: str) -> float:
        if self.avg_face is None or self.eigen_faces is None:
            raise ValueError('Model has not been trained')
        with open(csv_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            img_representations, names, img_file_paths = [], [], []
            for img_path, name in csv_reader:
                try:
                    rep = self.get_representation(img_path)
                except:
                    continue
                img_representations.append(rep.flatten())
                names.append(name)
                img_file_paths.append(img_path)
        pred_names = []
        for rep in img_representations:
            min_dist = None
            pred_name = None
            for name, class_rep in self.classes_representation:
                # X = np.vstack([rep, class_rep.flatten()])
                # d = pdist(X, 'mahalanobis')
                d = np.sum((class_rep.flatten() - rep) ** 2)
                if min_dist is None or min_dist > d:
                    min_dist = d
                    pred_name = name
            pred_names.append(pred_name)

        right_num = 0
        for i in range(len(names)):
            if names[i] != pred_names[i]:
                pass
                # print(f'pred: {pred_names[i]}, gt: {names[i]}')
            else:
                right_num += 1
        acc = right_num / len(names)

        self.result_writer.write(image_paths=img_file_paths,
                                 pred_names=pred_names,
                                 names=names,
                                 accuracy=acc,
                                 fname=data_type + '.csv')
        return acc

    @staticmethod
    def _createDataMatrix(images):
        print("Creating data matrix", end=" ... ")
        ''' 
        Allocate space for all images in one data matrix. 
            The size of the data matrix is
            ( w  * h  * 3,  numImages )
    
            where,
            w = width of an image in the dataset.
            h = height of an image in the dataset.
            3 is for the 3 color channels.
        '''

        numImages = len(images)
        sz = images[0].shape
        data = np.zeros((numImages, sz[0] * sz[1] * sz[2]), dtype=np.float32)
        for i in range(0, numImages):
            image = images[i].flatten()
            data[i, :] = image

        print("DONE")
        return data.T

    @staticmethod
    def _PCA(matrix: np.ndarray, PCA_size=10):
        avg_face = np.average(matrix, axis=1)
        A = matrix.T - avg_face
        A = A.T
        eigen_values, eigen_vectors = np.linalg.eigh(np.dot(A.T, A))
        sorted_indices = np.argsort(eigen_values)
        top_indices = sorted_indices[:-PCA_size - 1:-1]
        eigen_vectors = eigen_vectors[top_indices]

        ret_eigen_vectors = [cv2.normalize(src=np.dot(A, eigen_vector).T, dst=None, norm_type=cv2.NORM_L2) for
                             eigen_vector in
                             eigen_vectors]
        return avg_face, ret_eigen_vectors

    @staticmethod
    def get_eigen_face(image_paths: List[str], eigen_size: int = 100, transform=None):
        """
        Generate eigen face for image
        :param eigen_size: size for eigen vector after PCA
        :param image_paths: the list of  paths of images to process
        :param transform: transformation on image. Default is None.
        :return: eigen faces and average face
        """
        images = []
        print('Read images...')
        for image_path in tqdm(image_paths):
            assert os.path.exists(image_path), f'invalid image path: {image_path}'
            try:
                img = Image.open(image_path)
                img = img.convert('RGB')
            except:
                continue
            if transform is not None:
                img = transform(img)
            img = np.array(img)
            images.append(img)

        shape = images[0].shape

        img_data = EigenFaceRecognizer._createDataMatrix(images)

        print('Calculate PCA', end=' ... ')
        mean, eigen_vectors = EigenFaceRecognizer._PCA(img_data, eigen_size)
        print('DONE')

        avg_face = mean.reshape(shape)

        print('Get eigen faces for images', end=' ... ')
        eigen_faces = []
        for eigen_vector in eigen_vectors:
            # eigen_face = eigen_vector.reshape(shape)
            eigen_faces.append(eigen_vector)
        print('DONE')

        return eigen_faces, avg_face

    def save_model(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({
                'classes_representation': self.classes_representation,
                'avg_face': self.avg_face,
                'eigen_faces': self.eigen_faces
            }, f)

    def load_model(self, pkl_path: str):
        with open(pkl_path, 'rb') as f:
            checkpoint = pickle.load(f)
            self.classes_representation = checkpoint['classes_representation']
            self.avg_face = checkpoint['avg_face']
            self.eigen_faces = checkpoint['eigen_faces']


def eigen(config: Config, train: bool) -> None:
    print('eigen face recognition start ... ')
    transform = transforms.Compose([
        transforms.Resize(config.img_size)
    ])

    os.makedirs(os.path.join(config.checkpoint_folder, 'eigen_faces'), exist_ok=True)
    dataset_names = config.dataset_names
    for dataset_name in dataset_names:
        print(f'{dataset_name} begin...')
        eigen_recognizer = EigenFaceRecognizer(transform=transform, face_detect=config.detect_face, config=config)
        checkpoint_path = os.path.join(config.checkpoint_folder, 'eigen_faces', config.config_name,
                                       f'{dataset_name}_model.pkl')
        if train:
            train_csv_path = os.path.join(config.csv_folder, dataset_name, 'train.csv')
            eigen_recognizer.train(train_csv_path)
            eigen_recognizer.save_model(checkpoint_path)
            train_acc = eigen_recognizer.predict(train_csv_path, f'{dataset_name}_train')
            print(f'train_acc: {train_acc * 100: .3f}%')

        eigen_recognizer.load_model(checkpoint_path)

        test_acc = eigen_recognizer.predict(os.path.join(config.csv_folder, dataset_name, 'test.csv'), f'{dataset_name}_test')
        print(f'test_acc: {test_acc * 100: .3f}%')
        print()
