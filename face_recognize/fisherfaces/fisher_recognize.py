from face_recognize.fisherfaces.fisher_faces import PCALDAClassifier
from torchvision.transforms import transforms
import os
from config.Config import Config
from results.ResultWriter import ResultWriter


def fisher(config: Config, train: bool):
    print('fisher face recognition start ... ')
    transform = transforms.Compose([
        transforms.Resize(config.img_size)
    ])

    os.makedirs(os.path.join(config.checkpoint_folder, 'fisher_faces'), exist_ok=True)
    dataset_names = config.dataset_names
    for dataset_name in dataset_names:
        print(f'{dataset_name} begin...')
        fisher_recognizer = PCALDAClassifier(transform=transform, config=config)
        checkpoint_path = os.path.join(config.checkpoint_folder, 'fisher_faces', config.config_name,
                                       f'{dataset_name}_model.pkl')

        if train:
            train_csv_path = os.path.join(config.csv_folder, dataset_name, 'train.csv')
            fisher_recognizer.train_csv_file(train_csv_path)
            fisher_recognizer.save_model(checkpoint_path)
            train_acc = fisher_recognizer.predict_csv_file(train_csv_path, f'{dataset_name}_train')
            print(f'train_acc: {train_acc * 100: .3f}%')

        fisher_recognizer.load_model(checkpoint_path)

        test_acc = fisher_recognizer.predict_csv_file(os.path.join(config.csv_folder, dataset_name, 'test.csv'),
                                                      f'{dataset_name}_test')
        print(f'test_acc: {test_acc * 100: .3f}%')
        print()
