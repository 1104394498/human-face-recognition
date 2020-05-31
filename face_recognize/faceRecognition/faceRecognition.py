import face_recognition as fr
from sklearn import svm
import os
import pickle
from tqdm import tqdm
import csv
from config.Config import Config
from results.ResultWriter import ResultWriter


class FaceRecognizer:
    def __init__(self, config: Config):
        self.clf = svm.SVC(gamma='scale')
        self.result_writer = ResultWriter(config=config, method='face_rec')

    def train(self, train_csv_path: str):
        assert os.path.exists(train_csv_path), f'invalid csv file path: {train_csv_path}'

        encodings = []
        names = []

        with open(train_csv_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            print('get image encoding...')
            for img_file_path, person in tqdm(csv_reader):
                try:
                    face = fr.load_image_file(img_file_path)
                    face_enc = fr.face_encodings(face)[0]
                except:
                    continue

                encodings.append(face_enc)
                names.append(person)
        print('start to fit...')
        self.clf.fit(encodings, names)

    def save_model(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(self.clf, f)

    def load_model(self, pkl_path: str):
        with open(pkl_path, 'rb') as f:
            self.clf = pickle.load(f)

    def predict_img(self, img_path: str) -> str:
        assert os.path.exists(img_path), f'invalid image path: {img_path}'
        img = fr.load_image_file(img_path)
        img_enc = fr.face_encodings(img)[0]
        name = self.clf.predict([img_enc])
        return name

    def predict(self, csv_file_path: str, data_type: str) -> float:
        assert os.path.exists(csv_file_path), f'invalid csv file path: {csv_file_path}'
        print('get image encoding...')
        right_num = 0
        encoding, names, img_file_paths = [], [], []
        with open(csv_file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for img_file_path, person in tqdm(csv_reader):
                try:
                    face = fr.load_image_file(img_file_path)
                    face_enc = fr.face_encodings(face)[0]
                except:
                    continue
                encoding.append(face_enc)
                img_file_paths.append(img_file_path)
                names.append(person)
        pred_names = self.clf.predict(encoding)
        for i in range(len(pred_names)):
            if pred_names[i] == names[i]:
                right_num += 1
            else:
                print(f'pred: {pred_names[i]}, gt: {names[i]}')

        acc = right_num / len(pred_names)
        self.result_writer.write(image_paths=img_file_paths,
                                 pred_names=pred_names,
                                 names=names,
                                 accuracy=acc,
                                 fname=data_type + '.csv')
        return acc


def face_rec(config: Config, train: bool):
    os.makedirs(os.path.join(config.checkpoint_folder, 'face_rec'), exist_ok=True)
    dataset_names = config.dataset_names
    for dataset_name in dataset_names:
        print(f'{dataset_name} begin...')
        recognizer = FaceRecognizer(config)
        checkpoint_path = os.path.join(config.checkpoint_folder, 'face_rec', config.config_name,
                                       f'{dataset_name}_model.pkl')

        if train:
            train_csv_path = os.path.join(config.csv_folder, dataset_name, 'train.csv')
            recognizer.train(train_csv_path)
            recognizer.save_model(checkpoint_path)
            train_acc = recognizer.predict(train_csv_path, f'{dataset_name}_train')
            print(f'test_acc: {train_acc * 100: .3f}%')

        recognizer.load_model(checkpoint_path)
        test_acc = recognizer.predict(os.path.join(config.csv_folder, dataset_name, 'test.csv'), f'{dataset_name}_test')
        print(f'test_acc: {test_acc * 100: .3f}%')
        print()
