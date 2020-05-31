import yaml
import os


class Config:
    def __init__(self):
        self.config_name = ''
        self.img_size = (200, 180)
        self.dataset_names = ['faces94', 'faces95', 'faces96', 'grimace']
        self.detect_face = True
        self.checkpoint_folder = 'checkpoints'
        self.csv_folder = 'datasets'
        self.result_folder = 'results'

    @staticmethod
    def from_yaml(yaml_path: str):
        assert os.path.exists(yaml_path), f'invalid yaml config path: {yaml_path}'
        with open(yaml_path, 'r') as f:
            yaml_contents = yaml.load(f, Loader=yaml.FullLoader)
        config = Config()
        for k, v in yaml_contents.items():
            assert hasattr(config, k)
            if isinstance(v, list):
                v = tuple(v)
            setattr(config, k, v)
        return config


if __name__ == '__main__':
    config = Config.from_yaml('yaml_files/config_no_face_detect.yaml')
