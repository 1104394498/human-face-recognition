import numpy as np
from scipy.spatial import distance
from face_detect.detect_face import FaceDetector
import pickle
import os
from typing import Dict
import csv
from PIL import Image
from torchvision.transforms import transforms
from typing import Tuple, List
from results.ResultWriter import ResultWriter
from config.Config import Config


class Projection:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.subspace_basis = None

    def fit(self, X, y):
        """Fit the projection onto the training data."""

    def project(self, X):
        """Project the new data using the fitted projection matrices."""

    def reconstruct(self, X):
        """Reconstruct the projected data back into the original space."""

    def _check_fitted(self):
        """Check that the projector has been fitted."""
        assert self.subspace_basis is not None, \
            'You must fit %s before you can project' % self.__class__.__name__

    @property
    def P(self):
        self._check_fitted()
        return self.subspace_basis[:, :self.n_components]


class PCA(Projection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X_mean = None
        self.eigenvalues = None

    def fit(self, X, y=None):
        assert X.ndim == 2, 'X can only be a 2-d matrix'

        # Center the data
        self.X_mean = np.mean(X, axis=0)
        X = X - self.X_mean

        # If the d >> n then we should use dual PCA for efficiency
        use_dual_pca = X.shape[1] > X.shape[0]

        if use_dual_pca:
            X = X.T

        # Estimate the covariance matrix
        C = np.dot(X.T, X) / (X.shape[0] - 1)

        U, S, V = np.linalg.svd(C)

        if use_dual_pca:
            U = X.dot(U).dot(np.diag(1 / np.sqrt(S * (X.shape[0] - 1))))

        self.subspace_basis = U
        self.eigenvalues = S

        return self

    def project(self, X):
        self._check_fitted()
        X = X - self.X_mean
        return np.dot(X, self.P)

    def reconstruct(self, X):
        self._check_fitted()
        return np.dot(X, self.P.T) + self.X_mean


class LDA(Projection):
    def __init__(self, auto_components=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eigenvalues = None
        self.class_means = None
        self.auto_components = auto_components

    def fit(self, X, y):
        assert X.shape[0] == y.shape[0], 'X and y dimensions do not match.'

        n_classes = np.max(y) + 1
        n_samples, n_features = X.shape

        if self.auto_components:
            self.n_components = n_classes - 1
        else:
            assert self.n_components <= n_classes, \
                'LDA has (c - 1) non-zero eigenvalues. ' \
                'Please change n_components to <= '

        # Compute the class means
        class_means = np.zeros((n_classes, n_features))
        for i in range(n_classes):
            class_means[i, :] = np.mean(X[y == i], axis=0)

        mean = np.mean(class_means, axis=0)

        Sw, Sb = 0, 0
        for i in range(n_classes):
            # Compute the within class scatter matrix
            for j in X[y == i]:
                val = np.atleast_2d(j - class_means[i])
                Sw += np.dot(val.T, val)

            # Compute the between class scatter matrix
            val = np.atleast_2d(class_means[i] - mean)
            Sb += n_samples * np.dot(val.T, val)

        # Get the eigenvalues and eigenvectors in ascending order
        eigvals, eigvecs = np.linalg.eig(np.dot(np.linalg.inv(Sw), Sb))
        sorted_idx = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[sorted_idx], eigvecs[:, sorted_idx]

        self.subspace_basis = eigvecs.astype(np.float64)
        self.eigenvalues = eigvals

        self.class_means = np.dot(class_means, self.P)

        return self

    def project(self, X):
        self._check_fitted()
        return np.dot(X, self.P)

    def reconstruct(self, X):
        self._check_fitted()
        return np.dot(X, self.P.T)


class PCALDA(Projection):
    def __init__(self, pca_components=25, n_components=2):
        super().__init__(n_components)
        self.pca_components = pca_components
        self.pca = None
        self.lda = None

    def fit(self, X, y):
        self.pca = PCA(n_components=self.pca_components).fit(X)
        projected = self.pca.project(X)
        self.lda = LDA(n_components=self.n_components).fit(projected, y)

        self.subspace_basis = np.dot(self.pca.P, self.lda.P)

        return self

    def project(self, X):
        self._check_fitted()
        return np.dot(X - self.pca.X_mean, self.subspace_basis)

    def reconstruct(self, X):
        return X.dot(self.lda.P.T).dot(self.pca.P.T) + self.pca.X_mean


class Classifier:
    def train(self, X, y):
        """Fit the model on the given data."""

    def predict(self, X):
        """Return the labels for the given data."""


class PCALDAClassifier(Classifier):
    def __init__(self, config, pca_components=25, n_components=2, metric='euclidean', face_detect=True,
                 transform=None, img_size=(200, 180)):
        # type: (Config, int, int, str, bool, transforms.Compose, Tuple[int, int]) -> None
        self.pca_components = pca_components
        self.n_components = n_components
        self.metric = metric
        self.pca_lda = None
        self.label2name = None
        self.transform = transform

        if face_detect:
            self.face_detector = FaceDetector(
                prototxt='face_detect/checkpoints/deploy.prototxt.txt',
                checkpoint_path='face_detect/checkpoints/res10_300x300_ssd_iter_140000.caffemodel',
                img_size=img_size
            )
        else:
            self.face_detector = None
        self.result_writer = ResultWriter(config, 'fisher')

    def _get_image_data_from_csv(self, csv_path):
        # type: (str) -> Tuple[np.ndarray, np.ndarray, Dict[int:str], List[str]]
        assert os.path.exists(csv_path), f'invalid path: {csv_path}'

        csv_file = open(csv_path, 'r')
        csv_reader = csv.reader(csv_file)

        label2name, name2label = {}, {}
        images, labels, imgs_paths = [], [], []

        label = 0
        print('Reading images ... ', end='')
        for image_path, name in csv_reader:
            try:
                img = Image.open(image_path).convert('RGB')
                if self.face_detector is not None:
                    x_begin, x_end, y_begin, y_end = self.face_detector.detect(image_path, False)
                    img = img.crop((x_begin, y_begin, x_end, y_end))
                if self.transform is not None:
                    img = self.transform(img)
                img = np.array(img)
            except:
                continue
            imgs_paths.append(image_path)
            images.append(np.ravel(img))
            if name not in name2label:
                label2name[label] = name
                name2label[name] = label
                label += 1
            labels.append(name2label[name])

        csv_file.close()
        print('DONE')
        return np.vstack(images), np.hstack(labels), label2name, imgs_paths

    def train(self, X, y):
        # print(X.shape)
        print('start training ... ', end='')
        self.pca_lda = PCALDA(
            pca_components=self.pca_components,
            n_components=self.n_components,
        ).fit(X, y)
        print('DONE')
        return self

    def train_csv_file(self, csv_path: str):
        images, labels, label2name, _ = self._get_image_data_from_csv(
            csv_path=csv_path
        )
        self.train(images, labels)

    def predict(self, X, return_distances=False):
        assert self.pca_lda is not None, \
            'You must fit %s first' % self.__class__.__name__

        # Find the nearest class mean to each new sample
        class_means = self.pca_lda.lda.class_means

        projected = self.pca_lda.project(np.atleast_2d(X))

        distances = distance.cdist(projected, class_means, metric=self.metric)
        min_indices = np.argmin(distances, axis=1)

        if return_distances:
            return min_indices, distances
        return min_indices

    def predict_proba(self, X):
        indices, distances = self.predict(X, return_distances=True)
        # Perform softmax on negative distances because a good distance is a
        # low distance, and softmax does the inverse
        probs = softmax(-distances)
        return indices, probs

    def predict_csv_file(self, csv_path, data_type):
        # type: (str, str) -> float
        """
        :return: accuracy
        """
        assert os.path.exists(csv_path), f'invalid csv path: {csv_path}'
        assert self.pca_lda is not None, \
            'You must fit %s first' % self.__class__.__name__

        images, labels, label2name, img_paths = self._get_image_data_from_csv(csv_path)
        pred_indices = self.predict(images)
        # print(f'pred_indices len: {len(pred_indices)}, names len: {len(names)}')
        right_num = 0
        assert len(pred_indices) == len(labels)
        pred_names = [label2name[pred_indices[i]] for i in range(len(pred_indices))]
        names = [label2name[labels[i]] for i in range(len(labels))]
        for i in range(len(pred_names)):
            if pred_names[i] == names[i]:
                right_num += 1
        acc = right_num / len(pred_indices)
        self.result_writer.write(image_paths=img_paths,
                                 pred_names=pred_names,
                                 names=names,
                                 accuracy=acc,
                                 fname=data_type + '.csv')
        return acc

    def save_model(self, fname):
        dir_name = os.path.dirname(fname)
        os.makedirs(dir_name, exist_ok=True)
        face_detector = self.face_detector
        self.face_detector = None
        with open(fname, 'wb') as f:
            pickle.dump(self, f)
        self.face_detector = face_detector

    def load_model(self, fname):
        assert os.path.exists(fname), f'invalid file name: {fname}'
        with open(fname, 'rb') as f:
            checkpoint = pickle.load(f)
            self.pca_components = checkpoint.pca_components
            self.n_components = checkpoint.n_components
            self.metric = checkpoint.metric
            self.pca_lda = checkpoint.pca_lda
            self.label2name = checkpoint.label2name
            self.transform = checkpoint.transform


def softmax(X):
    X = np.atleast_2d(X)
    z = np.exp(X - np.max(X, axis=1, keepdims=True))
    probabilities = z / np.sum(z, axis=1, keepdims=True)
    return probabilities
