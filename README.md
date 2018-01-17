# CheXNet-with-localization
ADLxMLDS 2017 fall final

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

testing:
```
python3 denseNet_localization.py [path to test.txt] [path of images folder]
```


### Note :
In my .py script, I used the following script to assign the task running on GPU 0.<br>

```
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
```