import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os, sys
import pickle
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def compute_AUCs(gt, pred):
	
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


# ====== prepare dataset ======
class ChestXrayDataSet(Dataset):
    def __init__(self, train_or_valid = "train", transform=None):

        data_path = sys.argv[1]
        self.train_or_valid = train_or_valid
        if train_or_valid == "train":
            self.X = np.uint8(np.load(data_path + "train_X_small.npy")*255*255)
            with open(data_path + "train_y_onehot.pkl", "rb") as f:
                self.y = pickle.load(f)
            sub_bool = (self.y.sum(axis=1)!=0)
            self.y = self.y[sub_bool,:]
            self.X = self.X[sub_bool,:]
        else:
            self.X = np.uint8(np.load(data_path + "valid_X_small.npy")*255*255)
            with open(data_path + "valid_y_onehot.pkl", "rb") as f:
                self.y = pickle.load(f)
        
        self.label_weight_pos = (len(self.y)-self.y.sum(axis=0))/len(self.y)
        self.label_weight_neg = (self.y.sum(axis=0))/len(self.y)
#         self.label_weight_pos = len(self.y)/self.y.sum(axis=0)
#         self.label_weight_neg = len(self.y)/(len(self.y)-self.y.sum(axis=0))
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item 
        Returns:
            image and its labels
        """
        current_X = np.tile(self.X[index],3) 
        label = self.y[index]
        label_inverse = 1- label
        weight = np.add((label_inverse * self.label_weight_neg),(label * self.label_weight_pos))
        if self.transform is not None:
            image = self.transform(current_X)
        return image, torch.from_numpy(label).type(torch.FloatTensor), torch.from_numpy(weight).type(torch.FloatTensor)
    def __len__(self):
        return len(self.y)

# construct model
class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x





if __name__ == '__main__':

	# prepare training set
	train_dataset = ChestXrayDataSet(train_or_valid="train",
                                    transform=transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.RandomCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                        ]))
	augment_img = []
	augment_label = []
	augment_weight = []
	for i in range(4):
		for j in range(len(train_dataset)):
			single_img, single_label, single_weight = train_dataset[j]
			augment_img.append(single_img)
			augment_label.append(single_label)
			augment_weight.append(single_weight)
			if j % 1000==0:
				print(j)

	# shuffe data
	perm_index = torch.randperm(len(augment_label))
	augment_img = torch.stack(augment_img)[perm_index]
	augment_label = torch.stack(augment_label)[perm_index]
	augment_weight = torch.stack(augment_weight)[perm_index]

	# prepare validation set
	valid_dataset = ChestXrayDataSet(train_or_valid="valid",
					transform=transforms.Compose([
							transforms.ToPILImage(),
							transforms.CenterCrop(224),
							transforms.ToTensor(),
							transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
							]))

	valid_loader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=False, num_workers=16)
	# ====== start trianing =======

	cudnn.benchmark = True
	N_CLASSES = 8
	BATCH_SIZE = 64

	# initialize and load the model
	model = DenseNet121(N_CLASSES).cuda()
	model = torch.nn.DataParallel(model).cuda()

	optimizer = optim.Adam(model.parameters(),lr=0.0002, betas=(0.9, 0.999))
	total_length = len(augment_img)
	for epoch in range(10):  # loop over the dataset multiple times
		print("Epoch:",epoch)
		running_loss = 0.0

		# shuffle
		perm_index = torch.randperm(len(augment_label))
		augment_img = augment_img[perm_index]
		augment_label = augment_label[perm_index]
		augment_weight = augment_weight[perm_index]

		for index in range(0, total_length , BATCH_SIZE):
			if index+BATCH_SIZE > total_length:
				break
			# zero the parameter gradients
			optimizer.zero_grad()
			inputs_sub = augment_img[index:index+BATCH_SIZE]
			labels_sub = augment_label[index:index+BATCH_SIZE]
			weights_sub = augment_weight[index:index+BATCH_SIZE]
			inputs_sub, labels_sub = Variable(inputs_sub.cuda()), Variable(labels_sub.cuda())
			weights_sub = Variable(weights_sub.cuda())

			# forward + backward + optimize
			outputs = model(inputs_sub)
			criterion = nn.BCELoss()
			loss = criterion(outputs, labels_sub)
			loss.backward()
			optimizer.step()
			running_loss += loss.data[0]


		# ======== validation ======== 
		# switch to evaluate mode
		model.eval()


		# initialize the ground truth and output tensor
		gt = torch.FloatTensor()
		gt = gt.cuda()
		pred = torch.FloatTensor()
		pred = pred.cuda()


		for i, (inp, target, weight) in enumerate(valid_loader):
			target = target.cuda()
			gt = torch.cat((gt, target), 0)
			#     bs, n_crops, c, h, w = inp.size()
			input_var = Variable(inp.view(-1, 3, 224, 224).cuda(), volatile=True)
			output = model(input_var)
			#     output_mean = output.view(bs, n_crops, -1).mean(1)
			pred = torch.cat((pred, output.data), 0)

		CLASS_NAMES = ['Atelectasis', 'Cardiomegaly','Effusion', 'Infiltration',
						'Mass','Nodule', 'Pneumonia', 'Pneumothorax']

		AUROCs = compute_AUCs(gt, pred)
		AUROC_avg = np.array(AUROCs).mean()
		print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
		for i in range(N_CLASSES):
		    print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))

		model.train()
		# print statistics
		print('[%d] loss: %.3f' % (epoch + 1, running_loss / 715 ))
		torch.save(model.state_dict(),'DenseNet121_aug4_pretrain_noWeight_'+str(epoch+1)+'_'+str(AUROC_avg)+'.pkl')

	print('Finished Training')