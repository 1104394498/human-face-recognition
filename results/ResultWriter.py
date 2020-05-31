from config.Config import Config
import csv
from typing import List
import os


class ResultWriter:
    def __init__(self, config: Config, method: str):
        self.config = config
        self.method = method

    def write(self, image_paths: List[str], pred_names: List[str], names: List[str], accuracy: float, fname: str):
        csv_path = os.path.join(self.config.result_folder, self.method, self.config.config_name, fname)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        assert len(image_paths) == len(pred_names) and len(image_paths) == len(names)
        with open(csv_path, 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(('accuracy', accuracy))
            csv_writer.writerow(())
            csv_writer.writerow(('image path', 'predicted name', 'real name'))
            for i in range(len(image_paths)):
                csv_writer.writerow((image_paths[i], pred_names[i], names[i]))
