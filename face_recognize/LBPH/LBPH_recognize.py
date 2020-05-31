from face_recognize.LBPH.lbph import LBPHRecognizer
from torchvision.transforms import transforms
from config.Config import Config
import os


def lbph(config: Config, train: bool):
    print('LBPH recognition start ... ')

    transform = transforms.Compose([
        transforms.Resize(config.img_size)
    ])

    recognizer = LBPHRecognizer(transform=transform, face_detect=config.detect_face, config=config)

    dataset_names = config.dataset_names
    for dataset_name in dataset_names:
        print(f'{dataset_name} begin...')
        checkpoint_path = os.path.join(config.checkpoint_folder, 'LBPH', config.config_name,
                                       f'{dataset_name}_model.pkl')
        if train:
            train_csv_path = os.path.join(config.csv_folder, dataset_name, 'train.csv')
            recognizer.train(train_csv_path)
            recognizer.save_model(checkpoint_path)
            train_acc = recognizer.predict(train_csv_path, f'{dataset_name}_train')
            print(f'train_acc: {train_acc * 100 :.3f}%')
        recognizer.load_model(checkpoint_path)
        test_acc = recognizer.predict(os.path.join(config.csv_folder, dataset_name, 'test.csv'), f'{dataset_name}_test')
        print(f'test_acc: {test_acc * 100: .3f}%')
        print()
