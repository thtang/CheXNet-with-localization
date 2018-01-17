# CheXNet-with-localization
ADLxMLDS 2017 fall final

Team:XD

黃晴(R06922014),王思傑(R06922019),曹爗文(R06922022),傅敏桓(R06922030),湯忠憲(R06946003)
### Package : 
`Pytorch` &nbsp; `torchvision` &nbsp;` matplotlib`  &nbsp;` sklearn-image` &nbsp;

### usage:
preprocessing:
```
python3 preprocessing.py [path of images folder] [path to data_entry] [path to bbox_list_path] [path to train_txt] [path to valid_txt] [path of preprocessed output (folder)]
```

training:
```
python3 train.py [path of preprocessed output (folder)]
```

local testing:
```
python3 denseNet_localization.py [path to test.txt] [path of images folder]
```

DeepQueue testing:

1) upload **deepQ_82.zip** to the platform
2)
```
python3 inference.py
```


### Note :
In my .py script, I used the following script to assign the task running on GPU 0.<br>

```
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
```
### result :
![Alt Text](https://github.com/thtang/CheXNet-with-localization/blob/master/bb_select.JPG)
