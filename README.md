# human-face-recognition
# 如何运行

## 初次运行

首先下载[数据集](https://cswww.essex.ac.uk/mv/allfaces/index.html)。

初次运行时，需要对数据集进行随机划分，分成70\%的训练集和30\%的测试集（由于采用cross-validation，所以没有划分专门的验证集），并将划分结果存储在csv文件中：

```bash
python main.py config/yaml_files/{yaml_file_name} -m {method} -d -p {dataset_path} -t
```

其中`{yaml_file_name}`可以为`config_no_face_detect.yaml`或者`config_with_face_detect.yaml`，分别对应无face detection操作和有face detection操作的设定；

`{method}`可以为`eigen`, `fisher`,  `lbph`和`face_rec`，分别对应`eigenfaces`, `fisherfaces`, `LBPH`和`HOC`的算法；

`{dataset_path}`为[数据集](https://cswww.essex.ac.uk/mv/allfaces/index.html)存放的文件夹。

## 需要重新训练

```bash
python main.py config/yaml_files/{yaml_file_name} -m {method} -t
```

其中`{yaml_file_name}`可以为`config_no_face_detect.yaml`或者`config_with_face_detect.yaml`，分别对应无face detection操作和有face detection操作的设定；

`{method}`可以为`eigen`, `fisher`,  `lbph`和`face_rec`，分别对应`eigenfaces`, `fisherfaces`, `LBPH`和`HOC`的算法；

`-t`表示需要训练。

## 不需要重新训练

```
python main.py config/yaml_files/{yaml_file_name} -m {method}
```

其中`{yaml_file_name}`可以为`config_no_face_detect.yaml`或者`config_with_face_detect.yaml`，分别对应无face detection操作和有face detection操作的设定；

`{method}`可以为`eigen`, `fisher`,  `lbph`和`face_rec`，分别对应`eigenfaces`, `fisherfaces`, `LBPH`和`HOC`的算法。
