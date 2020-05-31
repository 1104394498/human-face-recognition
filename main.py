import argparse
from config.Config import Config
from datasets.divide_dataset import divide_dataset
from face_recognize.eigen_faces.eigen_faces import eigen
from face_recognize.faceRecognition.faceRecognition import face_rec
from face_recognize.fisherfaces.fisher_recognize import fisher
from face_recognize.LBPH.LBPH_recognize import lbph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='path of yaml config file')
    parser.add_argument('-m', '--method', type=str, default='face_rec', help='method for face recognization')
    parser.add_argument('-d', '--divide', action='store_true', help='divide dataset or not')
    parser.add_argument('-p', '--path', type=str, default='',
                        help='Path of dataset to divide. If args.divide is False, this can be empty')
    parser.add_argument('-t', '--train', action='store_true', help='train model or not')
    args = parser.parse_args()

    config = Config.from_yaml(args.config)

    if args.divide:
        divide_dataset(dataset_path=args.path, output_dir=config.csv_folder)

    if args.method == 'face_rec':
        face_rec(config, args.train)
    elif args.method == 'eigen':
        eigen(config, args.train)
    elif args.method == 'fisher':
        fisher(config, args.train)
    elif args.method == 'lbph':
        lbph(config, args.train)
    else:
        raise ValueError(f'invalid method: {args.method}')
