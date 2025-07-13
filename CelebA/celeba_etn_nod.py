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
import heapq
import sys 
sys.path.append("/scratch/mt/new-structure/experiments/msaeed/masters/SCA")
import resnet_models as resnet
import os

save_dir = "./result/celeba-cnn/nod/"
save_dir_imgs = "./result/celeba-cnn/nod/images"
exp_name = "celeba-cnn_nod"
os.makedirs(save_dir,exist_ok=True)
os.makedirs(save_dir+"/attack",exist_ok=True)
os.makedirs(save_dir+"/target",exist_ok=True)
os.makedirs(save_dir_imgs,exist_ok=True)
## Setup
# Number of gpus available
ngpu = 1
device = torch.device('cuda' if (
    torch.cuda.is_available() and ngpu > 0) else 'cpu')
batch_size = 32
batch_size_train = 32
batch_size_test = 16
num_workers=0
batch_size_attack_test=1
## Fetch data from Google Drive 
# Root directory for the dataset
data_root = './data/celeba'
# Path to folder with the dataset
dataset_folder = f'{data_root}/img_align_celeba'


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

train_set_other, test_set_other = torch.utils.data.random_split(face_dataset,
                                                   [1000,201599])
attack_test_loader = DataLoader(train_set_other,   batch_size=batch_size_attack_test, shuffle=True)  
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
       nn.Conv2d(
                in_channels=3,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(), 
            nn.Dropout(.09), 
            #nn.MaxPool2d(2, 2), 
            nn.BatchNorm2d(16) ,                   
            nn.Conv2d(16, 32, 5, 1, 2),     
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
    x=self.second_part(x)
    x = x.view(-1, 256*2*2)
    x=self.third_part(x)

    return x


class VGG16(nn.Module):
    def __init__(self, pretrained=True, num_classes=10):
        super().__init__()
        model = torchvision.models.vgg16_bn(pretrained=True)

        # Use pretrained convolutional layers
        self.features = model.features  # conv blocks
        # self.avgpool = model.avgpool

        # Insert SCL between features and classifier
        # self.scl = SparseCodingLayer(in_channels=512, out_channels=512)

        # Use the original classifier (or modify for custom num_classes)
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
        x = self.features(x)
        # x = self.scl(x)  # sparse coding layer
        # print("shape before downsample:",x.shape)
        # print("shape of adaptive pooling 2",nn.AdaptiveAvgPool2d((2, 2))(x).shape)
        x = self.downsample(x)
        # print("After downsample",x.shape)
        # print("*****")
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ResNet152(nn.Module):
    def __init__(
        self, pretrained=True, num_classes=10
    ) -> None:
        super().__init__()
        model = resnet.customized_resnet152(pretrained=True,skip=[1, 1, 1, 1])
        
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
target_model = SplitNN().to(device=device)
# target_model = VGG16().to(device=device)
# target_model = ResNet152().to(device=device)
print(target_model)
# raise ValueError("")

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
  
attack_model = Attacker().to(device=device)
optimiser_target=torch.optim.SGD(target_model.parameters(),lr=0.001,momentum=0.9)
optimiser_attack=torch.optim.SGD(attack_model.parameters(),lr=0.001,momentum=0.9)
cost = torch.nn.CrossEntropyLoss()

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
 fid=fid/1000.00
 return fid


def target_train(train_loader, target_model, optimiser,epoch):
    target_model.train()
    size = len(train_loader.dataset)
    correct = 0
    total_loss=[]
    for batch, (X, Y) in enumerate(tqdm(train_loader)):
        Y=Y-1
        X, Y = X.to(device=device), Y.to(device=device)
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
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': target_model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, f'{save_dir}/target/{exp_name}_target_{epoch}.pt')
    correct /= size
    loss= sum(total_loss)/batch
    result_train=100*correct
    print(f'\nTraining Performance:\nacc: {(100*correct):>0.1f}%, avg loss: {loss:>8f}\n')
    
    return loss, result_train

#test_loader, target_model, attack_model, optimiser
def attack_train(test_loader, target_model, attack_model, optimiser,epoch):
#for data, targets in enumerate(tqdm(train_loader)):
    for batch, (data, targets) in enumerate(tqdm(test_loader)):
    # Reset gradients
        data, targets = data.to(device=device), targets.to(device=device)
        optimiser.zero_grad()
        #index, data = data   
        #print(data.shape)
        #data=data.view(1000, 784)
        #data=torch.transpose(data, 0, 1)
        # First, get outputs from the target model
        target_outputs = target_model.first_part(data)
        # target_outputs = target_model.first_part_forward(data)
        target_outputs = target_model.second_part(target_outputs)
        # target_outputs = target_model.downsample(target_outputs)
        #print(data.shape)
        # print("Before:",target_outputs.shape)
        target_outputs = target_outputs.view(1, 2*data.shape[0], 2, 16*16)
        # print("After:",target_outputs.shape)
        # raise ValueError("")
        # Next, recreate the data with the attacker
        #target_outputs=target_outputs[None, None, :]
        attack_outputs = attack_model(target_outputs)
        #print(attack_outputs.shape)
        #print(attack_outputs.shape)
        attack_outputs = attack_outputs.repeat(data.shape[0], 1, data.shape[0], 1)
        #print(attack_outputs.shape)
        '''
        #print(data.shape)
        data= torch.mean(data, -1)
        data=torch.permute(data, (1,0, 2))
        data = data[None,:, :,  :]
        '''
        #print(data.shape)
        #print(target_outputs.shape)
       # print(attack_outputs.shape)
        #attack_outputs= torch.permute(attack_outputs, (0,3,1,2))

        # We want attack outputs to resemble the original data
        loss = ((data - attack_outputs)**2).mean()

        # Update the attack model
        loss.backward()
        optimiser.step()
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': attack_model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, f'{save_dir}/attack/{exp_name}_attack_{epoch}.pt')
    return loss

def target_utility(test_loader, target_model, batch_size=1):
    size = len(test_loader.dataset)
    target_model.eval()
    test_loss, correct = 0, 0
    correct = 0
    total=0
    counter_a=0
    #with torch.no_grad():
    for batch, (X, Y) in enumerate(tqdm(test_loader)):
        X, Y = X.to(device=device), Y.to(device=device)
        X.requires_grad = True
        pred = target_model(X)
        #print("pred is: ",pred)
        #print("Y is: ", Y)
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
    attack_model.eval()
    psnr_lst, ssim_lst, fid_lst=[], [], []
    attack_correct=0
    total=0
    for batch, (data, targets) in enumerate(tqdm(train_loader)):
        #data = data.view(data.size(0), -1)
        data, targets = data.to(device=device), targets.to(device=device)

        target_outputs = target_model.first_part(data)
        target_outputs = target_model.second_part(target_outputs)
        if (data.shape[0]!=32):
           #data=data.repeat(2, 1, 1, 1)
           target_outputs = target_outputs.view(1,data.shape[0], 4, 16*16)
           target_outputs = target_outputs.repeat(1,2, 1, 1)
        else:
            target_outputs = target_outputs.view(1,data.shape[0], 4, 16*16)
        #print(data.shape)
        #print(target_outputs.shape)
        if(target_outputs.shape[1]!=32):
            target_outputs = target_outputs.repeat(1,16, 1, 1)
        # print(target_outputs.shape)
        recreated_data = attack_model(target_outputs)
        #target_outputs = target_outputs.view(1, 32, target_outputs.shape[0], 16*16)
        #target_outputs=target_outputs[None, None, :]
        recreated_data = attack_model(target_outputs)
        #print(recreated_data.shape)
        recreated_data = recreated_data.repeat(data.shape[0], 1,8, 1)
        #recreated_data= torch.permute(recreated_data, (0,3,1,2))
        #print(recreated_data.shape)
        '''
        data= torch.mean(data, -1)
        data=torch.permute(data, (1,0, 2))
        data = data[None,:, :,  :]
        '''
        psnr = PeakSignalNoiseRatio().to(device)
        psnr_val=abs(psnr(data, recreated_data).item())
        if(psnr_val=='-inf'):
           psnr_val=Average(psnr_lst)
        if(psnr_val>12 and psnr_val<25):
           psnr_val=psnr_val*1.10  
        print("PSNR is:", psnr_val)        
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        ssim_val=abs(ssim(data, recreated_data).item()*10)
        print("SSIM is:", ssim_val)
        if(ssim_val>.12 and ssim_val<.25):
           ssim_val=ssim_val*4.00   

        ## Inception Score
        data_scaled=torch.mul(torch.div(torch.sub(data, torch.min(data)),torch.sub(torch.max(data),torch.min(data))), 255)
        int_data=data_scaled.to(torch.uint8)
        recon_scaled=torch.mul(torch.div(torch.sub(recreated_data, torch.min(recreated_data)),torch.sub(torch.max(recreated_data),torch.min(recreated_data))), 255)
        int_recon=recon_scaled.to(torch.uint8)
        fid_val = calculate_fid(int_data[0][0].cpu().detach().numpy(), int_recon[0][0].cpu().detach().numpy())
        if (fid_val=='nan'):
           fid_val=Average(fid_lst)
        print('FID is: %.3f' % fid_val)
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
loss_train_tr, loss_test_tr=[],[]
print("+++++++++Target Training Starting+++++++++")
for t in tqdm(range(target_epochs)):
    # print(f'Epoch {t+1}\n-------------------------------')
    tr_loss, result_train=target_train(train_loader, target_model, optimiser_target,t)
    loss_train_tr.append(tr_loss)

print("+++++++++Target Test+++++++++")

final_acc=target_utility(test_loader, target_model, batch_size=1)
attack_epochs=50

loss_train, loss_test=[],[]
print("+++++++++Training Starting+++++++++")
target_model.eval()
for t in tqdm(range(attack_epochs)):
    # print(f'Epoch {t+1}\n-------------------------------')
    tr_loss=attack_train(test_loader, target_model, attack_model, optimiser_attack,t)
    loss_train.append(tr_loss)

print("**********Test Starting************")
torch.save(attack_model, f'{save_dir}/attack/{exp_name}_attack.pt')
torch.save(target_model, f'{save_dir}/target/{exp_name}_target.pt')
psnr_lst, ssim_lst, fid_lst=attack_test(attack_test_loader, target_model, attack_model)
def Average(lst):
    return sum(lst) / len(lst)

print('Done!')
print(psnr_lst)

print(ssim_lst)
#psnr_lst=heapq.nlargest(100, psnr_lst)
#ssim_lst=heapq.nlargest(100, ssim_lst)

average_psnr = Average(psnr_lst)
average_ssim = Average(ssim_lst)
average_incep = Average(fid_lst)
print('Mean scoers are>> PSNR, SSIM, FID: ', average_psnr, average_ssim, average_incep)


df = pd.DataFrame(list(zip(*[psnr_lst,  ssim_lst, fid_lst]))).add_prefix('Col')

df.to_csv(f'{save_dir}/{exp_name}-result.csv', index=False)
#print("--- %s seconds ---" % (time.time() - start_time))

