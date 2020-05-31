import csv
import os
import numpy as np
from PIL import Image
from typing import Tuple, Dict
from tqdm import tqdm
from torchvision.transforms import transforms


def get_image_data_from_csv(csv_path, transform=None):
    # type: (str, transforms.Compose) -> Tuple[np.ndarray, np.ndarray, Dict[int:str]]
    assert os.path.exists(csv_path), f'invalid path: {csv_path}'

    csv_file = open(csv_path, 'r')
    csv_reader = csv.reader(csv_file)

    label2name, name2label = {}, {}
    images, labels = [], []

    label = 0
    print('Reading images ... ')
    for image_path, name in tqdm(csv_reader):
        try:
            img = Image.open(image_path).convert('RGB')
            if transform is not None:
                img = transform(img)
            img = np.array(img)
        except:
            continue
        images.append(np.ravel(img))
        if name not in name2label:
            label2name[label] = name
            name2label[name] = label
            label += 1
        labels.append(name2label[name])

    csv_file.close()
    print('DONE')
    return np.vstack(images), np.hstack(labels), label2name
