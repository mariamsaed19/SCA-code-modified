import os
import zipfile 
import gdown
import torch
import torchvision
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch import nn, optim
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.inception import InceptionScore
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import numpy as np
import torch.nn.functional as F
from numpy import iscomplexobj
from numpy import cov
from numpy import trace
from scipy.linalg import sqrtm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from skimage import io
from lcapt.lca import LCAConv2D
import sys
sys.path.append("/scratch/mt/new-structure/experiments/msaeed/masters/SCA")
import resnet_models as resnet
## Setup
# Number of gpus available
ngpu = 1
device = torch.device('cuda' if (
    torch.cuda.is_available() and ngpu > 0) else 'cpu')
batch_size = 32
batch_size_train = 8
batch_size_test = 16
num_workers=0

## Fetch data from Google Drive 
# Root directory for the dataset
data_root = './data/celeba'
# Path to folder with the dataset
dataset_folder = f'{data_root}/img_align_celeba/'


class FaceCelebADataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks_bald = self.landmarks_frame.iloc[idx, 5]
        landmarks_black = self.landmarks_frame.iloc[idx, 9]
        landmarks_blond = self.landmarks_frame.iloc[idx, 10]
        landmarks_brown = self.landmarks_frame.iloc[idx, 12]
        landmarks_gray = self.landmarks_frame.iloc[idx, 18]
            
        landmarks=abs(landmarks_bald+landmarks_black+ landmarks_blond + landmarks_brown+ landmarks_gray)
        #landmarks = self.landmarks_frame.iloc[idx, 1]
        y_label = torch.tensor(int(landmarks))
        #landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
        sample = {'image': image, 'landmarks': y_label}
        

        if self.transform:
            image = self.transform(image)
        
        return (image, y_label)
        

transform__op=torchvision.transforms.Compose([
                                 #transforms.ToPILImage(),

                               torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Resize((32)),
                                                          transforms.CenterCrop(32),
                                                         transforms.Normalize(mean=[0.5, 0.5, 0.5],
                          std=[0.5, 0.5, 0.5])
                              ])

face_dataset = FaceCelebADataset(csv_file='./data/celeba/list_attr_celeba.csv',
                                    root_dir=dataset_folder, transform=transform__op)
#face_dataset = FaceLandmarksDataset(csv_file='/dartfs-hpc/rc/home/h/f0048vh/Sparse_guard/celeba/data/list_landmarks_align_celeba.csv',
#                                    root_dir=dataset_folder, transform=transform__op)
print(len(face_dataset))
train_set, test_set_org = torch.utils.data.random_split(face_dataset,
                                                   [50000,152599])
test_set, test_set2 = torch.utils.data.random_split(test_set_org,
                                                   [20000,132599])
train_loader = DataLoader(train_set,   batch_size=batch_size_train, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size_test, shuffle=True)

  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use gpu if available
#152624

print("Working Fine")
size1 = len(train_loader)
size2 = len(test_loader)

print(size1, size2)

class SplitNN(nn.Module):
  def __init__(self):
    super(SplitNN, self).__init__()
    self.first_part = nn.Sequential(
     LCAConv2D(out_neurons=16,
                in_neurons=3,                        
                kernel_size=5,              
                stride=1,  
                tau=1000,                  
                 lambda_=0.5, lca_iters=500, pad="same",               
            ),                               
            nn.ReLU(), 
            nn.Dropout(.09), 
            #nn.MaxPool2d(2, 2), 
            nn.BatchNorm2d(16) ,                   
    LCAConv2D(out_neurons=32,
                in_neurons=16,                        
                kernel_size=5,              
                stride=1,     
                tau=1000,               
                 lambda_=0.5, lca_iters=500, pad="same", ),    
                                         nn.ReLU(), 
            nn.Dropout(.09), 
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),                    

                         )
    self.second_part = nn.Sequential(
                           nn.Conv2d(32, 64, 5, 1, 2),     
                            nn.ReLU(), 
                            nn.Dropout(.09), 
                            nn.MaxPool2d(2, 2),
                            nn.BatchNorm2d(64),   
                            nn.Conv2d(64, 128, 5, 1, 2),     
                            nn.ReLU(), 
                            nn.Dropout(.09), 
                            nn.MaxPool2d(2, 2),
                            nn.BatchNorm2d(128), 
                            nn.Conv2d(128, 256, 5, 1, 2),     
                            nn.ReLU(), 
                            nn.Dropout(.09),  
                            nn.MaxPool2d(2, 2),
                            nn.BatchNorm2d(256),  
                            
                           #scancel nn.Softmax(dim=-1),
                         )
    self.third_part = nn.Sequential(
                            nn.Linear(256*2*2, 500),
                            nn.ReLU(),
                            nn.Linear(500, 5),
       

    )

  def forward(self, x):
    x=self.first_part(x)
    #print(x.shape)
    #x = torch.flatten(x, 1) # flatten all dimensions except batch
    #x = x.view(-1, 32*16*500)
    #print(x.shape)
    x=self.second_part(x)
    #print(x.shape)
    x = x.view(-1, 256*2*2)
    x=self.third_part(x)
    #print(x.shape)

    return x

class VGG16WithSCLFirst(nn.Module):
    def __init__(self, pretrained=True, num_classes=10):
        super().__init__()
        # base = models.vgg16(pretrained=pretrained)
        model = torchvision.models.vgg16_bn(pretrained=True)

        self.first_part = nn.Sequential(
            LCAConv2D(out_neurons=16,
                        in_neurons=3,                        
                        kernel_size=5,              
                        stride=1,     
                        tau=1000,               
                        lambda_=0.5, lca_iters=500, pad="same",               
                    ),                               
                    nn.ReLU(), 
                    nn.Dropout(.09), 
                    #nn.MaxPool2d(2, 2), 
                    nn.BatchNorm2d(16) ,                   
            LCAConv2D(out_neurons=32,
                        in_neurons=16,                        
                        kernel_size=5,              
                        stride=1, 
                        tau=1000,                   
                        lambda_=0.5, lca_iters=500, pad="same", ),    
                                                nn.ReLU(), 
                    nn.Dropout(.09), 
                    # nn.MaxPool2d(2, 2), # don't downsample from 32 --> 16
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 3, kernel_size=1)   # adapter                

                                )
        # Use pretrained convolutional layers
        self.features = model.features  # conv blocks
        self.downsample = nn.Sequential(
            # model.avgpool,
            nn.AdaptiveAvgPool2d((2, 2)),     # [16, 512, 7, 7] -> [16, 512, 2, 2]
            nn.Conv2d(512, 256, kernel_size=1)  # Reduce channels: 512 → 256
        )
        self.classifier = nn.Sequential(
                            nn.Linear(256*2*2, 512),
                            nn.ReLU(),
                            nn.Linear(512, 10),
            )

    def forward(self, x):
        x = self.first_part(x)
        x = self.features(x)
        x = self.downsample(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class ResNet152(nn.Module):
    def __init__(
        self, pretrained=True, num_classes=10
    ) -> None:
        super().__init__()
        model = resnet.customized_resnet152(pretrained=True,skip=[1, 1, 1, 1])
        self.first_part = nn.Sequential(
            LCAConv2D(out_neurons=16,
                        in_neurons=3,                        
                        kernel_size=5,              
                        stride=1,     
                        tau=1000,               
                        lambda_=0.5, lca_iters=500, pad="same",               
                    ),                               
                    nn.ReLU(), 
                    nn.Dropout(.09), 
                    #nn.MaxPool2d(2, 2), 
                    nn.BatchNorm2d(16) ,                   
            LCAConv2D(out_neurons=32,
                        in_neurons=16,                        
                        kernel_size=5,              
                        stride=1, 
                        tau=1000,                   
                        lambda_=0.5, lca_iters=500, pad="same", ),    
                                                nn.ReLU(), 
                    nn.Dropout(.09), 
                    # nn.MaxPool2d(2, 2), # don't downsample from 32 --> 16
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 3, kernel_size=1)   # adapter                

                                )
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.downsample = nn.Sequential(
            # model.avgpool,
            nn.AdaptiveAvgPool2d((2, 2)),     # [16, 512, 7, 7] -> [16, 512, 2, 2]
            nn.Conv2d(2048, 1024, kernel_size=1),  # Reduce channels: 512 → 256
            nn.Conv2d(1024, 512, kernel_size=1),  # Reduce channels: 512 → 256
            nn.Conv2d(512, 256, kernel_size=1)  # Reduce channels: 512 → 256
        )
        self.fc = nn.Sequential(
                            nn.Linear(256*2*2, 512),
                            nn.ReLU(),
                            nn.Linear(512, 10),
            )

    def first_part_forward(self,x):
        x = self.first_part(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.downsample(x)
        
        return x
    def forward(self, x):
        # See note [TorchScript super()]
        x = self.first_part_forward(x)
        # print("before classifier",x.shape)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
# target_model = SplitNN().to(device=device, dtype=torch.float16)
# target_model = VGG16WithSCLFirst().to(device=device, dtype=torch.float16)
# target_model = ResNet152().to(device=device, dtype=torch.float16)
# print(target_model)
class Attacker(nn.Module):
  def __init__(self):
    super(Attacker, self).__init__()
    self.layers= nn.Sequential(
                     nn.Linear(256, 512),
                      nn.ReLU(),
                      nn.Linear(512, 32),
                      nn.ReLU(),
                      nn.ConvTranspose2d(32, 16, 5, 1, 2, bias=False),
                      nn.BatchNorm2d(16),
                      nn.ReLU(),
                      nn.ConvTranspose2d(16, 10, 5, 1, 2, bias=False),
                      nn.BatchNorm2d(10),
                      nn.ReLU(),
                      nn.ConvTranspose2d(10, 3, 5, 1, 2, bias=False),

                    )
 
  def forward(self, x):
    return self.layers(x)
  
# attack_model = Attacker().to(device=device, dtype=torch.float16)
# optimiser=torch.optim.SGD(target_model.parameters(),lr=0.001,momentum=0.9)
# cost = torch.nn.CrossEntropyLoss()

# calculate frechet inception distance
def calculate_fid(act1, act2):
 # calculate mean and covariance statistics
 mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
 mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
 # calculate sum squared difference between means
 ssdiff = np.sum((mu1 - mu2)**2.0)
 # calculate sqrt of product between cov
 covmean = sqrtm(sigma1.dot(sigma2))
 # check and correct imaginary numbers from sqrt
 if iscomplexobj(covmean):
  covmean = covmean.real
 # calculate score
 fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
 return fid


def target_train(train_loader, target_model, optimiser):
    target_model.train()
    size = len(train_loader.dataset)
    correct = 0
    total_loss=[]
    for batch, (X, Y) in enumerate(tqdm(train_loader)):
        Y=Y-1
        X, Y = X.to(device=device, dtype=torch.float16), Y.to(device=device)
        #print(X, Y)
        target_model.zero_grad()
        pred = target_model(X)
        #print(pred.shape, Y.shape)
        loss = cost(pred, Y)
        loss.backward()
        optimiser.step()
        _, output = torch.max(pred, 1)
        correct+= (output == Y).sum().item()
        total_loss.append(loss.item())
        #batch_count+=batch
        #correct += (pred.argmax(1)==Y).type(torch.float).sum().item()

    correct /= size
    loss= sum(total_loss)/batch
    result_train=100*correct
    print(f'\nTraining Performance:\nacc: {(100*correct):>0.1f}%, avg loss: {loss:>8f}\n')
    
    return loss, result_train

#test_loader, target_model, attack_model, optimiser
def attack_train(test_loader, target_model, attack_model, optimiser):
#for data, targets in enumerate(tqdm(train_loader)):
    for batch, (data, targets) in enumerate(tqdm(test_loader)):
    # Reset gradients
        data, targets = data.to(device=device, dtype=torch.float16), targets.to(device=device)
        optimiser.zero_grad()

        target_outputs = target_model.first_part_forward(data)
        # target_outputs = target_model.features(target_outputs)
        # target_outputs = target_model.downsample(target_outputs)
        #print(data.shape)
        #print(target_outputs.shape)
        target_outputs = target_outputs.view(1, 2*data.shape[0], 2, 16*16)
        #print(target_outputs.shape)
        # Next, recreate the data with the attacker
        #target_outputs=target_outputs[None, None, :]
        attack_outputs = attack_model(target_outputs)
        #print(attack_outputs.shape)
        #print(attack_outputs.shape)
        attack_outputs = attack_outputs.repeat(data.shape[0], 1, data.shape[0], 1)
        #print(attack_outputs.shape)

        loss = ((data - attack_outputs)**2).mean()

        # Update the attack model
        loss.backward()
        optimiser.step()

    return loss

def target_utility(test_loader, target_model, batch_size=1):
    size = len(test_loader.dataset)
    #target_model.eval()
    test_loss, correct = 0, 0
    correct = 0
    total=0
    counter_a=0
    #with torch.no_grad():
    for batch, (X, Y) in enumerate(tqdm(test_loader)):
        X, Y = X.to(device=device, dtype=torch.float16), Y.to(device=device)
        X.requires_grad = True
        pred = target_model(X)
        counter_a=counter_a+1
        #test_loss += cost(pred, Y).item()
        #correct += (pred.argmax(1)==Y).type(torch.float).sum().item()


        #data, target = data.to(device), target.to(device)
       
        # Set requires_grad attribute of tensor. Important for Attack
        total += Y.size(0)
        # Forward pass the data through the model
        _, output_res = torch.max(pred, -1)
        correct += ((output_res+1) == Y).sum().item()


    # Calculate final accuracy for this epsilon
    final_acc = correct/float(total)
    print(f"Target Model Accuracy = {correct} / {total} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc 


def attack_test(train_loader, target_model, attack_model):
    psnr_lst, ssim_lst, fid_lst=[], [], []
    attack_correct=0
    total=0
    for batch, (data, targets) in enumerate(tqdm(train_loader)):
        #data = data.view(data.size(0), -1)
        data, targets = data.to(device=device, dtype=torch.float16), targets.to(device=device)
        target_outputs = target_model.first_part_forward(data)
        # target_outputs = target_model.features(target_outputs)
        # target_outputs = target_model.downsample(target_outputs)
        if (data.shape[0]!=32):
           #data=data.repeat(2, 1, 1, 1)
           target_outputs = target_outputs.view(1,data.shape[0], 4, 16*16)
           target_outputs = target_outputs.repeat(1,2, 1, 1)
        else:
            target_outputs = target_outputs.view(1,data.shape[0], 4, 16*16)

        if(target_outputs.shape[1]!=32):
            target_outputs = target_outputs.repeat(1,2, 1, 1)
        recreated_data = attack_model(target_outputs)
        #print(recreated_data.shape)
        recreated_data = recreated_data.repeat(data.shape[0], 1,8, 1)

        psnr = PeakSignalNoiseRatio().to(device)
        psnr_val=abs(psnr(data, recreated_data).item())
        if (psnr_val=='-inf'):
           psnr_val=Average(psnr_lst)
        print("PSNR is:", psnr_val)
        #print("PSNR is:", psnr_val)
        
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        ssim_val=abs(ssim(data, recreated_data).item())
        print("SSIM is:", ssim_val)


        ## Inception Score
        data_scaled=torch.mul(torch.div(torch.sub(data, torch.min(data)),torch.sub(torch.max(data),torch.min(data))), 255)
        int_data=data_scaled.to(torch.uint8)
        recon_scaled=torch.mul(torch.div(torch.sub(recreated_data, torch.min(recreated_data)),torch.sub(torch.max(recreated_data),torch.min(recreated_data))), 255)
        int_recon=recon_scaled.to(torch.uint8)
        fid_val = calculate_fid(int_data[0][0].cpu().detach().numpy(), int_recon[0][0].cpu().detach().numpy())
        if (fid_val=='nan'):
           fid_val=Average(fid_val)
        print("FID is:", fid_val)
        #print('FID is: %.3f' % fid_val)
        if (recreated_data.shape[2]==16):
           recreated_data=recreated_data.repeat(1, 1, 2, 1)
        test_output = target_model(recreated_data)
       
        _, pred = torch.max(test_output, -1)
        attack_correct += (pred == targets).sum().item()
        total += targets.size(0)

        psnr_lst.append(psnr_val)
        ssim_lst.append(ssim_val)
        fid_lst.append(fid_val)
    
    attack_acc = attack_correct/float(total)
    print(f" Attack Performance = {attack_correct} / {total} = {attack_acc}\t")

    return psnr_lst, ssim_lst, fid_lst

target_epochs=25

target_model = torch.load('./result-celeba/resnet152_CelebA_lca2_epoch_CNN_cnn_target.pt', map_location='cpu',weights_only=False)
attack_model = torch.load('./result-celeba/resnet152_CelebA_lca2_epoch_CNN_cnn_attack.pt', map_location='cpu',weights_only=False)
# Wrap the models

target_model.to(device)
attack_model.to(device)
loss_train_tr, loss_test_tr=[],[]
# print("+++++++++Target Training Starting+++++++++")
# for t in tqdm(range(target_epochs)):
#     # print(f'Epoch {t+1}\n-------------------------------')
#     tr_loss, result_train=target_train(train_loader, target_model, optimiser)
#     loss_train_tr.append(tr_loss)

print("+++++++++Target Test+++++++++")

final_acc=target_utility(test_loader, target_model, batch_size=1)
attack_epochs=50

loss_train, loss_test=[],[]
# print("+++++++++Training Starting+++++++++")
# for t in tqdm(range(attack_epochs)):
#     # print(f'Epoch {t+1}\n-------------------------------')
#     tr_loss=attack_train(test_loader, target_model, attack_model, optimiser)
#     loss_train.append(tr_loss)

print("**********Test Starting************")
# torch.save(attack_model, './result-celeba/resnet152_CelebA_lca2_epoch_CNN_cnn_attack.pt')
# torch.save(target_model, './result-celeba/resnet152_CelebA_lca2_epoch_CNN_cnn_target.pt')
psnr_lst, ssim_lst, fid_lst=attack_test(train_loader, target_model, attack_model)
def Average(lst):
    return sum(lst) / len(lst)

print('Done!')


average_psnr = Average(psnr_lst)
average_ssim = Average(ssim_lst)
average_incep = Average(fid_lst)
print('Mean scoers are>> PSNR, SSIM, FID: ', average_psnr, average_ssim, average_incep)


df = pd.DataFrame(list(zip(*[psnr_lst,  ssim_lst, fid_lst]))).add_prefix('Col')

df.to_csv('./result-celeba/resnet152_CelebA_lca2_epoch_CNN_attack_cnn.csv', index=False)

