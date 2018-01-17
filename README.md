# CheXNet-with-localization
ADLxMLDS 2017 fall final

### Package : 
`Pytorch` &nbsp; `torchvision` &nbsp;` matplotlib`  &nbsp;` sklearn-image` &nbsp;

### usage:

```
python3 denseNet_localization.py [path to test.txt] [path of images folder]
```


### Note :
In my .py script, I used the following script to assign the task running on GPU 0.<br>

```
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
```