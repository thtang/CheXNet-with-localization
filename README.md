# CheXNet-with-localization
ADLxMLDS 2017 fall final

Team:XD

*黃晴 (R06922014), 王思傑 (R06922019), 曹爗文 (R06922022), 傅敏桓 (R06922030), 湯忠憲 (R06946003)*
## Weakly supervised localization :
![Alt Text](https://github.com/thtang/CheXNet-with-localization/blob/master/output/process_flow.png)

### Package : 
`Pytorch==0.2.0` &nbsp; `torchvision==0.2.0` &nbsp;` matplotlib`  &nbsp;` scikit-image==0.13.1` &nbsp;` opencv_python==3.4.0.12` &nbsp;` numpy==1.13.3` &nbsp;`matplotlib==2.1.1` &nbsp;`scipy==1.0.0` &nbsp; `sklearn==0.19.1` &nbsp;

### Environment:
* OS: Linux
* Python 3.5
* GPU: 1080 ti
* CPU: Xeon(R) E5-2667 v4
* RAM: 500 GB
### Experiments process:
1) preprocessing:
```
python3 preprocessing.py [path of images folder] [path to data_entry] [path to bbox_list_path] [path to train_txt] [path to valid_txt] [path of preprocessed output (folder)]
```

2) training:
```
python3 train.py [path of preprocessed output (folder)]
```

3) local testing:
```
python3 denseNet_localization.py [path to test.txt] [path of images folder]
```

4) DeepQ testing:

upload **deepQ_25.zip** to the platform. Then use following command:
```
python3 inference.py
```


### Note :
In our .py script, I used the following script to assign the task running on GPU 0.<br>

```
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
```
### Result :
*Prediction*<br>
<img src="https://github.com/thtang/CheXNet-with-localization/blob/master/output/prediction.png" width="240"><br>
*Heatmap per disease*
![Alt Text](https://github.com/thtang/CheXNet-with-localization/blob/master/output/heatmap_per_class.jpg)
Visualization of some heat maps with its ground-truth label (red) and its prediction
(blue) selected from each disease class. (From top-left to bottom: Atelectasis, Cardiomegaly,
Effusion, Infiltration, Mass, Nodule, Pneumonia and Pneumothorax)

*Bounding Box per patient*
![Alt Text](https://github.com/thtang/CheXNet-with-localization/blob/master/output/bb_select.JPG)
Visualization of some images with its ground-truth label (red) and its prediction
(blue) selected from each disease class.

## Reference:
1. [ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases](https://arxiv.org/pdf/1705.02315.pdf)
2. [LEARNING TO DIAGNOSE FROM SCRATCH BY EXPLOITING DEPENDENCIES AMONG LABELS](https://arxiv.org/pdf/1710.10501.pdf)
3. [CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning](https://arxiv.org/pdf/1711.05225.pdf)
4. [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/pdf/1610.02391.pdf)
