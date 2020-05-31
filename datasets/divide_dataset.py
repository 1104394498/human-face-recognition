import os
import random
import csv


def divide_dataset(
        dataset_path: str,
        output_dir: str,
        train_ratio: float = 0.7,
        random_seed: int = 7
):
    """
    将数据集2（https://cswww.essex.ac.uk/mv/allfaces/index.html）的图片划分为训练集与测试集
    :param: dataset_path: 数据集的路径
    :param: train_ratio: 训练集所占比例
    :param: val_ratio: 验证集所占比例
    :param: random_seed: 随机种子
    :return: None
    """

    def divide_photos(person_path: str, train_writer, test_writer):
        photos = [photo for photo in os.listdir(person_path) if
                  len(photo) > 0 and photo[0] != '.']
        random.shuffle(photos)
        train_num = int(train_ratio * len(photos))
        train_photos = photos[:train_num]
        test_photos = photos[train_num:]

        for photo in train_photos:
            file_path = os.path.abspath(os.path.join(cur_person_path, photo))
            person = photo.split('.')[0]
            train_writer.writerow((file_path, person))

        for photo in test_photos:
            file_path = os.path.abspath(os.path.join(cur_person_path, photo))
            person = photo.split('.')[0]
            test_writer.writerow((file_path, person))

    assert os.path.exists(dataset_path), f'invalid dataset path: {dataset_path}'
    os.makedirs(output_dir, exist_ok=True)
    random.seed(random_seed)

    test_ratio = 1 - train_ratio
    assert test_ratio > 0, f'invalid test_ratio: {test_ratio}'

    dataset_names = ['faces94', 'faces95', 'faces96', 'grimace']

    for dataset_name in dataset_names:
        os.makedirs(os.path.join(output_dir, dataset_name), exist_ok=True)
        train_file = open(os.path.join(output_dir, dataset_name, 'train.csv'), 'w')
        train_writer = csv.writer(train_file)

        test_file = open(os.path.join(output_dir, dataset_name, 'test.csv'), 'w')
        test_writer = csv.writer(test_file)
        if dataset_name == 'faces94':
            cur_dataset_path = os.path.join(dataset_path, dataset_name)
            kinds = [kind for kind in os.listdir(cur_dataset_path) if len(kind) > 0 and kind[0] != '.']
            for kind in kinds:
                kind_path = os.path.join(cur_dataset_path, kind)
                people = [person for person in os.listdir(kind_path) if len(person) > 0 and person[0] != '.']
                for person in people:
                    cur_person_path = os.path.join(kind_path, person)
                    divide_photos(cur_person_path, train_writer, test_writer)
        else:
            cur_dataset_path = os.path.join(dataset_path, dataset_name)
            people = os.listdir(cur_dataset_path)
            for person in people:
                if len(person) <= 0 or person[0] == '.':
                    continue
                cur_person_path = os.path.join(cur_dataset_path, person)
                divide_photos(cur_person_path, train_writer, test_writer)
        train_file.close()
        test_file.close()


if __name__ == '__main__':
    divide_dataset('/Volumes/Samsung_T5/我的文档/大三下/图像处理与模式识别/datasets/FaceRecognitionData',
                   '/datasets')
