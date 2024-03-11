import torch
import torchvision
from tqdm import tqdm
import torch
import torchvision
from tqdm import tqdm
from torch import nn, optim
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.inception import InceptionScore
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torchvision.transforms as T
import pandas as pd
import math
import numpy as np
import os
import torch.nn.functional as F
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from skimage import io
from lcapt.lca import LCAConv2D
import heapq
import time
import random as rand
from numpy.random import random
from PIL import Image
import PIL.Image
import dnnlib
import legacy
import fnmatch
import sys
import pickle
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torchvision.transforms as T
import pandas as pd
import math
import numpy as np
import os
import torch.nn.functional as F
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from skimage import io
from lcapt.lca import LCAConv2D
import heapq
import time
start_time = time.time()
##Target and Attack Model Parameters
n_epochs = 3
batch_size_train = 32
batch_size_attack=1
batch_size_test = 1
batch_size_attack_test = 1

## Below are for GAN Training
gan_batch_size = 32
num_epochs = 25
n_critic = 5
display_step = 25
#attack_test_loader
###Data Loader for the Target and Attack train and Test
data_root = '/dartfs-hpc/rc/home/h/f0048vh/Sparse_guard/celeba/data'
# Path to folder with the dataset
dataset_folder = f'{data_root}/img_align_celeba/img_align_celeba'


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
        policy = self.landmarks_frame.iloc[idx, 0]
        image = io.imread(img_name)
        landmarks_bald = self.landmarks_frame.iloc[idx, 5]
        landmarks_black = self.landmarks_frame.iloc[idx, 9]
        landmarks_blond = self.landmarks_frame.iloc[idx, 10]
        landmarks_brown = self.landmarks_frame.iloc[idx, 12]
        landmarks_gray = self.landmarks_frame.iloc[idx, 18]
            
        landmarks=abs(landmarks_bald+landmarks_black+ landmarks_blond + landmarks_brown+ landmarks_gray)
        #landmarks = self.landmarks_frame.iloc[idx, 1]
        y_label = torch.tensor(int(landmarks)-1)
        #policy =  self.landmarks_frame[idx]
        #landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
        sample = {'image': image, 'landmarks': y_label}
        

        if self.transform:
            image = self.transform(image)
        
        return (image, y_label, policy)
        

transform__op=torchvision.transforms.Compose([
                                 #transforms.ToPILImage(),

                               torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Resize((32)),
                                                          transforms.CenterCrop(32),
                                                         transforms.Normalize(mean=[0.5, 0.5, 0.5],
                          std=[0.5, 0.5, 0.5])
                              ])

face_dataset = FaceCelebADataset(csv_file='/dartfs-hpc/rc/home/h/f0048vh/Sparse_guard/celeba/data/list_attr_celeba.csv',
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
train_loader2 = DataLoader(train_set_other,   batch_size=batch_size_train, shuffle=True)
data_loader = DataLoader(train_set,   batch_size=gan_batch_size, shuffle=True)
stylegan_test = FaceCelebADataset(csv_file='/dartfs-hpc/rc/home/h/f0048vh/Sparse_guard/celeba/data/test_attr.csv',
                                    root_dir=dataset_folder, transform=transform__op)
#batch_size_attack_test = 1
train_set_other, test_set_other = torch.utils.data.random_split(face_dataset,
                                                   [1000,201599])
attack_test_loader = DataLoader(stylegan_test,   batch_size=batch_size_attack_test, shuffle=True)  
  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use gpu if available
## Data Laoder for GAN Model
'''
data_loader = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR10('/dartfs-hpc/rc/home/h/f0048vh/Sparse_guard/data', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])),
  batch_size=gan_batch_size, shuffle=True)
'''

'''
attack_loader = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR10('/vast/home/sdibbo/def_ddlc/data', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])),
  batch_size=batch_size_attack, shuffle=True)
'''  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use gpu if available
##GAN  model
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.label_emb = nn.Embedding(6, 6)
        
        self.model = nn.Sequential(
            nn.Linear(3078, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        #print(x.shape)
        x = x.view(x.size(0), 3072)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        out = self.model(x)
        return out.squeeze()
    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.label_emb = nn.Embedding(6, 6)
        
        self.model = nn.Sequential(
            nn.Linear(42, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 3072),
            #nn.Tanh()
        )
    
    def forward(self, z, labels):
        #print(z.shape)
        z = z.view(z.size(0), 36)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        #print(x.shape)
        #x=x.view(-1, 110)
        out = self.model(x)
        #x=x.view(-1, 32*32*3)
        #print(x.shape)
        return out.view(x.size(0), 3, 32, 32)
        
generator = Generator().to(device)
discriminator = Discriminator().to(device)

d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
cost = torch.nn.CrossEntropyLoss()

def generator_train_step(batch_size, discriminator, generator, g_optimizer, cost):
    g_optimizer.zero_grad()
    z = Variable(torch.randn(batch_size, 36)).cuda()
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 6, batch_size))).cuda()
    fake_images = generator(z, fake_labels)
    validity = discriminator(fake_images, fake_labels)
    g_loss = cost(validity, Variable(torch.ones(batch_size)).cuda())
    g_loss.backward()
    g_optimizer.step()
    return g_loss.item()

def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, cost, real_images, labels):
    d_optimizer.zero_grad()

    # train with real images
    real_validity = discriminator(real_images, labels)
    real_loss = cost(real_validity, Variable(torch.ones(batch_size)).cuda())
    
    # train with fake images
    z = Variable(torch.randn(batch_size, 36)).cuda()
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 6, batch_size))).cuda()
    fake_images = generator(z, fake_labels)
    fake_validity = discriminator(fake_images, fake_labels)
    fake_loss = cost(fake_validity, Variable(torch.zeros(batch_size)).cuda())
    
    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.item()
def Average(lst):
  return sum(lst) / len(lst)
all_d_loss=[]
#no_len = (i for i in range(num_epochs))
for epoch in range(num_epochs):
    print('Starting epoch {}...'.format(epoch), end=' ')
    for i, (images, labels, idx) in enumerate (tqdm(data_loader)):
        
        step = epoch * len(data_loader) + i + 1
        real_images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        generator.train()
        

        d_loss = 0
        for _ in range(n_critic):
            d_loss = discriminator_train_step(len(real_images), discriminator,
                                              generator, d_optimizer, cost,
                                              real_images, labels)


        g_loss = generator_train_step(gan_batch_size, discriminator, generator, g_optimizer, cost)
        
       # writer.add_scalars('scalars', {'g_loss': g_loss, 'd_loss': (d_loss / n_critic)}, step)  
        
        if step % display_step == 0:
            generator.eval()
            z = Variable(torch.randn(5, 36)).cuda()
            labels = Variable(torch.LongTensor(np.arange(5))).cuda()
            sample_images = generator(z, labels).unsqueeze(1)
            #grid = make_grid(sample_images, nrow=3, normalize=True)
            #writer.add_image('sample_image', grid, step)
    all_d_loss.append(d_loss)
        
norm_d_loss=Average(all_d_loss)
print('GAN Training is Done!')


def generate_img(generator, digit):
    z = Variable(torch.randn(1, 36)).cuda()
    label = torch.LongTensor([digit]).cuda()
    img = generator(z, label).data.cpu()
    #img = 0.5 * img + 0.5
    return img
## Target and Attack Models

## Target and Attack Models
class Pub_classifier(nn.Module):
  def __init__(self):
    super(Pub_classifier, self).__init__()
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


class VictimNN(nn.Module):
  def __init__(self):
    super(VictimNN, self).__init__()
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


target_model = VictimNN().to(device=device, dtype=torch.float16)
clf_model = Pub_classifier().to(device=device, dtype=torch.float16)


#target_model = SplitNN().to(device=device)
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
                      nn.ConvTranspose2d(16, 5, 5, 1, 2, bias=False),
                      nn.BatchNorm2d(5),
                      nn.ReLU(),
                      nn.ConvTranspose2d(5, 3, 5, 1, 2, bias=False),

                    )
 
  def forward(self, x):
    return self.layers(x)
  
attack_model = Attacker().to(device=device)
optimiser=torch.optim.SGD(target_model.parameters(),lr=0.001,momentum=0.9)
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
 fid=math.sqrt(fid)
 return fid

def Average(lst):
    return sum(lst) / len(lst)

def poincare_loss(outputs, targets, xi=1e-4):
    # Normalize logits
    u = outputs / torch.norm(outputs, p=1, dim=-1).unsqueeze(0)
    # Create one-hot encoded target vector
    v = torch.clip(
        torch.eye(outputs.shape[-1], device=outputs.device)[targets] - xi, 0,
        1)
    v = v.to(u.device)
    # Compute squared norms
    u_norm_squared = torch.norm(u, p=2, dim=-1)**2
    v_norm_squared = torch.norm(v, p=2, dim=-1)**2
    diff_norm_squared = torch.norm(u - v, p=2, dim=-1)**2
    # Compute delta
    delta = 2 * diff_norm_squared / ((1 - u_norm_squared) *
                                     (1 - v_norm_squared))
    # Compute distance
    loss = torch.arccosh(1 + delta)
    return loss



def classifier_train(train_loader2, clf_model, optimiser):
    clf_model.train()
    size = len(train_loader2.dataset)
    correct = 0
    loss=0
    total_loss=[]
    for batch, (X, Y, idx) in enumerate(tqdm(train_loader2)):
        #print(target_model)
        #Y=Y-1
        X, Y = X.to(device=device, dtype=torch.float16), Y.to(device)
        #print(X.shape)
        target_model.zero_grad()
        pred = target_model(X)
        gan_loss=[]
        for i in range(Y[0]):
            label=Y[i]
            recreated_data=generate_img(generator, label)
            #print(recreated_data.shape)
            recon_img=recreated_data[0]/ 1 + 0.5
            #print(pred.shape, Y.shape)
            #recon_img=torch.permute(recon_img, (2,1, 0))
            recon_img=recon_img.unsqueeze(0).repeat(64, 1, 1, 1)
            #print(recon_img.shape)
            recon_img=recon_img.to(device=device, dtype=torch.float16)
            pred2 = target_model(recon_img)
            #print(label)
            #print(pred2[0])
            prob_max, output = torch.max(pred2[0], 0)
            #print(prob_max)
            loss2 = cost(pred2[0], label)
            if loss2.item()==0:
               loss.item=0.00001
            #print(loss2)
            _, output2 = torch.max(pred2[0], 0)
            correct+= (output2 == Y[i]).sum().item()
            gan_loss.append(loss2.item())
        #print(gan_loss)
        if not gan_loss:
           gan_loss.append(0.00001)
        loss_other=Average(gan_loss)    
        loss = cost(pred, Y)
        loss=loss+loss_other
        loss.backward()
        optimiser.step()
        _, output = torch.max(pred, 1)
        correct+= (output == Y).sum().item()
        total_loss.append(loss.item())
        #batch_count+=batch
        #correct += (pred.argmax(1)==Y).type(torch.float).sum().item()

    correct /= (2*size)
    loss= sum(total_loss)/batch
    result_train=100*correct
    print(f'\nClassifier Training Performance:\nacc: {(100*correct):>0.1f}%, avg loss: {loss:>8f}\n')
    
    return loss, result_train

def target_train(train_loader1, target_model, optimiser):
    target_model.train()
    size = len(train_loader1.dataset)
    correct = 0
    total_loss=[]
    for batch, (X, Y, idx) in enumerate(tqdm(train_loader1)):
        #print(target_model)
        #Y=Y-1
        X, Y = X.to(device=device, dtype=torch.float16), Y.to(device)
        #print(X.shape)
        target_model.zero_grad()
        pred = target_model(X)
        gan_loss=[]
        for i in range(Y[0]):
            label=Y[i]
            recreated_data=generate_img(generator, label)
            #print(recreated_data.shape)
            recon_img=recreated_data[0]/ 1 + 0.5
            #print(pred.shape, Y.shape)
            #recon_img=torch.permute(recon_img, (2,1, 0))
            recon_img=recon_img.unsqueeze(0).repeat(64, 1, 1, 1)
            #print(recon_img.shape)
            recon_img=recon_img.to(device=device, dtype=torch.float16)
            pred2 = target_model(recon_img)
            #print(label)
            #print(pred2[0])
            prob_max, output = torch.max(pred2[0], 0)
            #print(prob_max)
            loss2 = cost(pred2[0], label)
            if loss2.item()==0:
               loss.item=0.00001
            #print(loss2)
            _, output2 = torch.max(pred2[0], 0)
            correct+= (output2 == Y[i]).sum().item()
            gan_loss.append(loss2.item())
        #print(gan_loss)
        if not gan_loss:
           gan_loss.append(0.00001)
        loss_other=Average(gan_loss)    
        loss = cost(pred, Y)
        loss=loss+loss_other
        loss.backward()
        optimiser.step()
        _, output = torch.max(pred, 1)
        correct+= (output == Y).sum().item()
        total_loss.append(loss.item())
        #batch_count+=batch
        #correct += (pred.argmax(1)==Y).type(torch.float).sum().item()


    correct /= (2*size)
    loss= sum(total_loss)/batch
    result_train=100*correct
    print(f'\nTarget Victim Training Performance:\nacc: {(100*correct):>0.1f}%, avg loss: {loss:>8f}\n')
    
    return loss, result_train


'''
def target_train(train_loader, target_model, optimiser):
    target_model.train()
    size = len(train_loader.dataset)
    correct = 0
    total_loss=[]
    for batch, (X, Y) in enumerate(tqdm(train_loader)):
        #print(target_model)
        X, Y = X.to(device=device), Y.to(device)
        #print(X.shape)
        target_model.zero_grad()
        pred = target_model(X)
        gan_loss=[]
        for i in range(Y[0]):
            label=Y[i]
            recreated_data=generate_img(generator, label)
            #print(recreated_data.shape)
            recon_img=torch.permute(recreated_data[0]/ 2 + 0.5, (1,2, 0))
            #print(pred.shape, Y.shape)
            recon_img=torch.permute(recon_img, (2,1, 0))
            recon_img=recon_img.unsqueeze(0).repeat(64, 1, 1, 1)
            #print(recon_img.shape)
            recon_img=recon_img.to(device=device)
            pred2 = target_model(recon_img)
            loss2 = cost(pred2[0], Y[i])
            if loss2.item()==0:
               loss.item=0.00001
            #print(loss2)
            _, output2 = torch.max(pred2[0], 0)
            correct+= (output2 == Y[i]).sum().item()
            gan_loss.append(loss2.item())
        #print(gan_loss)
        if not gan_loss:
           gan_loss.append(0.00001)
        loss_other=Average(gan_loss)    
        loss = cost(pred, Y)
        loss=loss+loss_other
        loss.backward()
        optimiser.step()
        _, output = torch.max(pred, 1)
        correct+= (output == Y).sum().item()
        total_loss.append(loss.item())
        #batch_count+=batch
        #correct += (pred.argmax(1)==Y).type(torch.float).sum().item()

    correct /= (2*size)
    loss= sum(total_loss)/batch
    result_train=100*correct
    print(f'\nTraining Performance:\nacc: {(100*correct):>0.1f}%, avg loss: {loss:>8f}\n')
    
    return loss, result_train
'''
def clip_images(imgs):
    lower_limit = torch.tensor(-1.0).float().to(imgs.device)
    upper_limit = torch.tensor(1.0).float().to(imgs.device)
    imgs = torch.where(imgs > upper_limit, upper_limit, imgs)
    imgs = torch.where(imgs < lower_limit, lower_limit, imgs)
    return imgs
#test_loader, target_model, attack_model, optimiser
#test_loader, target_model, clf_model, attack_model, optimiser
def attack_train(test_loader, target_model, clf_model, attack_model, optimiser):
    model = VictimNN()
#for data, targets in enumerate(tqdm(train_loader)):
    generator.train()
    size = len(train_loader.dataset)
    correct = 0
    total_loss=[]
    #optimiser=torch.optim.Adam(generator.parameters(), lr=1e-4)
    for batch, (X, Y, idx) in enumerate(tqdm(test_loader)):
    # Reset gradients
        X, Y = X.to(device=device, dtype=torch.float16), Y.to(device=device)
        optimiser.zero_grad()
        pred = target_model(X)
        gan_loss=[]
        for i in range(1):
            label=Y[0]
            recreated_data=generate_img(generator, label)
            #print(recreated_data.shape)
            recon_img=torch.permute(recreated_data[0]/ 2 + 0.5, (1,2, 0))
            #print(pred.shape, Y.shape)
            recon_img=torch.permute(recon_img, (2,1, 0))
            #recon_img=recon_img.unsqueeze(0).repeat(1, 1, 1, 1)
            #print(recon_img.shape)
            recon_img=recon_img.to(device=device)
            recon_img=clip_images(recon_img)
            transform1=torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),torchvision.transforms.RandomResizedCrop(size=(32, 32),
                                scale=(0.5, 0.9),
                                ratio=(0.8, 1.2),
                                antialias=True),torchvision.transforms.ToTensor()])
            transform2=torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),torchvision.transforms.RandomHorizontalFlip(0.5),torchvision.transforms.ToTensor()])
            
            transform3=torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),torchvision.transforms.RandomHorizontalFlip(0.1),torchvision.transforms.ToTensor()])
            recreated_data1=transform1(recon_img)
            recreated_data2=transform2(recon_img)
            recreated_data3=transform3(recon_img)
            recon_img1=recreated_data1.unsqueeze(0).repeat(1, 1, 1, 1)
            recon_img2=recreated_data2.unsqueeze(0).repeat(1, 1, 1, 1)
            recon_img3=recreated_data3.unsqueeze(0).repeat(1, 1, 1, 1)


            outputs1 =  target_model(recon_img1.to(device=device, dtype=torch.float16)) # obtain logits
            outputs2 =  target_model(recon_img2.to(device=device, dtype=torch.float16)) # obtain logits
            outputs3=  target_model(recon_img3.to(device=device, dtype=torch.float16)) # obtain logits


            pred2 = target_model(recon_img.unsqueeze(0).repeat(1, 1, 1, 1).to (device=device, dtype=torch.float16))
            target_loss1 = poincare_loss(
            outputs1[0], Y[0]).mean()
            #print(target_loss)
            loss2 = cost(pred2[0], Y[0])
            if target_loss1.item()==0:
               loss.item=0.00001
            #print(loss2)
            _, output2 = torch.max(pred2[0], 0)
            correct+= (output2 == Y[0]).sum().item()
            gan_loss.append(target_loss1.item())
        #print(gan_loss)
        if not gan_loss:
           gan_loss.append(0.00001)
        loss = cost(pred, Y)
        target_loss1 = poincare_loss(
         outputs1[0], Y[0]).mean()
        target_loss2 = poincare_loss(
            outputs2[0], Y[0]).mean()
        target_loss3 = poincare_loss(
            outputs3[0], Y[0]).mean()
        loss_tar=target_loss1+target_loss2+ target_loss3+norm_d_loss
        pred = clf_model(X)
        gan_loss=[]
        for i in range(1):
            label=Y[0]
            recreated_data=generate_img(generator, label)
            #print(recreated_data.shape)
            recon_img=torch.permute(recreated_data[0]/ 2 + 0.5, (1,2, 0))
            #print(pred.shape, Y.shape)
            recon_img=torch.permute(recon_img, (2,1, 0))
            #recon_img=recon_img.unsqueeze(0).repeat(1, 1, 1, 1)
            #print(recon_img.shape)
            recon_img=recon_img.to(device=device)
            recon_img=clip_images(recon_img)
            transform1=torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),torchvision.transforms.RandomResizedCrop(size=(32, 32),
                                scale=(0.5, 0.9),
                                ratio=(0.8, 1.2),
                                antialias=True),torchvision.transforms.ToTensor()])
            transform2=torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),torchvision.transforms.RandomHorizontalFlip(0.5),torchvision.transforms.ToTensor()])
            
            transform3=torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),torchvision.transforms.RandomHorizontalFlip(0.1),torchvision.transforms.ToTensor()])
            recreated_data1=transform1(recon_img)
            recreated_data2=transform2(recon_img)
            recreated_data3=transform3(recon_img)
            recon_img1=recreated_data1.unsqueeze(0).repeat(1, 1, 1, 1)
            recon_img2=recreated_data2.unsqueeze(0).repeat(1, 1, 1, 1)
            recon_img3=recreated_data3.unsqueeze(0).repeat(1, 1, 1, 1)


            outputs1 =  clf_model(recon_img1.to(device=device, dtype=torch.float16)) # obtain logits
            outputs2 =  clf_model(recon_img2.to(device=device, dtype=torch.float16)) # obtain logits
            outputs3=  clf_model(recon_img3.to(device=device, dtype=torch.float16)) # obtain logits


            pred2 = clf_model(recon_img.unsqueeze(0).repeat(1, 1, 1, 1).to (device=device, dtype=torch.float16))
            clf_loss1 = poincare_loss(
            outputs1[0], Y[0]).mean()
            #print(target_loss)
            loss2 = cost(pred2[0], Y[0])
            if clf_loss1.item()==0:
               loss.item=0.00001
            #print(loss2)
            _, output2 = torch.max(pred2[0], 0)
            correct+= (output2 == Y[0]).sum().item()
            gan_loss.append(clf_loss1.item())
        #print(gan_loss)
        if not gan_loss:
           gan_loss.append(0.00001)
        loss = cost(pred, Y)
        clf_loss1 = poincare_loss(
         outputs1[0], Y[0]).mean()
        clf_loss2 = poincare_loss(
            outputs2[0], Y[0]).mean()
        clf_loss3 = poincare_loss(
            outputs3[0], Y[0]).mean()
        loss_clf=clf_loss1+clf_loss2+ clf_loss3+norm_d_loss
        #print(loss)
        loss=loss_tar+loss_clf
        loss.backward()
        optimiser.step()
       
    return loss

'''
        #index, data = data   
        #data=data.view(1000, 784)
        #data=torch.transpose(data, 0, 1)
        # First, get outputs from the target model
        target_outputs = target_model.first_part(data)
        #print(target_outputs.shape)
        target_outputs = target_outputs.view(1, 32, 1, 16*16)
        #target_outputs= target_outputs.view(-1,32*32*500)
        #target_outputs = target_model.second_part(target_outputs)
        # Next, recreate the data with the attacker
        attack_outputs = attack_model(target_outputs)
        #data= data.view(-1,32*32*3)
        # We want attack outputs to resemble the original data
        loss = ((data - attack_outputs)**2).mean()

        # Update the attack model
        loss.backward()
        optimiser.step()

    return loss
'''

def target_utility(test_loader, target_model, batch_size=1):
    size = len(test_loader.dataset)
    #target_model.eval()
    test_loss, correct = 0, 0
    correct = 0
    counter_a=0
    total=0
    #with torch.no_grad():
    for batch, (X, Y, idx) in enumerate(tqdm(test_loader)):
        X, Y = X.to(device=device, dtype=torch.float16), Y.to(device)
        X.requires_grad = True
        #X= X.view(-1,32*32*3)
        pred = target_model(X)
        counter_a=counter_a+1
        #test_loss += cost(pred, Y).item()
        #correct += (pred.argmax(1)==Y).type(torch.float).sum().item()


        #data, target = data.to(device), target.to(device)
       
        # Set requires_grad attribute of tensor. Important for Attack
        total += Y.size(0)
        # Forward pass the data through the model
        _, output_res = torch.max(pred, -1)
        correct += ((output_res) == Y).sum().item()


    # Calculate final accuracy for this epsilon
    final_acc = correct/float(total)
    print(f"Target Model Accuracy = {correct} / {total} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc 



def attack_test(train_loader, target_model, attack_model):
    model = VictimNN()
    psnr_lst, ssim_lst, fid_lst=[], [], []
    correct=0
    attack_correct=0
    total=0
    org_data=[]
    prob_lst=[]
    for batch, (X, Y, idx) in enumerate(tqdm(train_loader)):
        #data = data.view(data.size(0), -1)
        X, Y = X.to(device=device, dtype=torch.float16), Y.to(device=device)
        pred = target_model(X)
        class_prob = torch.softmax(pred, dim=1)
        # get most probable class and its probability:
        class_prob, topclass = torch.max(class_prob, dim=1)
        print(idx, class_prob)
        #org_data=data
        #data= data.view(-1,32*32*3)
        '''
        target_outputs = target_model.first_part(data)
        target_outputs = target_outputs.view(1, 32, target_outputs.shape[0], 16*16)
        #target_outputs= target_outputs.view(-1,32*32*500)
        #target_outputs = target_model.second_part(target_outputs)
        #target_outputs = target_model(data)
        recreated_data = attack_model(target_outputs)
        recreated_data=torch.permute(recreated_data, (2,1, 0,3))
        recreated_data=recreated_data.repeat(1,1,32,1) 
        print(data.shape)
        #print(data.shape)
        #print(target_outputs.shape)
        print(recreated_data.shape)
        #data=data.resize(64, 3, 32, 32)
        #recreated_data=recreated_data.resize(64, 3, 32, 32)
        #print(recreated_data.shape)
        '''
        #for i in range(Y[0]):
        label=Y
        recreated_data=generate_img(generator, label)
        #print(recreated_data.shape)
        recon_img=torch.permute(recreated_data[0]/ 2 + 0.5, (1,2, 0))
        #print(pred.shape, Y.shape)
        recon_img=torch.permute(recon_img, (2,1, 0))
        recon_img=recon_img.unsqueeze(0).repeat(1, 1, 1, 1)
        #print(recon_img.shape)
        recon_img=recon_img.to(device=device)
        pred2 = target_model(recon_img.to(dtype=torch.float16, device=device))
        
        recreated_data=recon_img
        
        #batch_count+=batch
        #correct += (pred.argmax(1)==Y).type(torch.float).sum().item()
        psnr = PeakSignalNoiseRatio().to(device)
        psnr_val=abs(psnr(X, recreated_data).item())
        if(psnr_val=='-inf'):
           psnr_val=Average(psnr_lst)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        ssim_val=abs(ssim(X, recreated_data).item())
    
        ## Inception Score

        data_scaled=torch.mul(torch.div(torch.sub(X, torch.min(X)),torch.sub(torch.max(X),torch.min(X))), 255)
        int_data=data_scaled.to(torch.uint8)
        recon_scaled=torch.mul(torch.div(torch.sub(recreated_data, torch.min(recreated_data)),torch.sub(torch.max(recreated_data),torch.min(recreated_data))), 255)
        int_recon=recon_scaled.to(torch.uint8)
        #print(int_data, int_recon)
        fid_val = abs(calculate_fid(int_data[0][0].cpu().detach().numpy(), int_recon[0][0].cpu().detach().numpy()))
        if (fid_val=='nan'):
           fid_val=Average(fid_lst)
        #gen_data= recreated_data.view(-1,32*32*3)

        #test_output = target_model(recreated_data)
        #attack_pred = test_output.max(1, keepdim=True)[1] # get the index of the max log-probability
        #print(f"Done with sample: {counter_a}\ton epsilon={epsilon}")

        #if attack_pred.item() == targets.item():
        #    attack_correct += 1        
        _, pred = torch.max(pred2, -1)
        attack_correct += (pred == Y).sum().item()
        total += Y.size(0)

        DataI = X[0] / 2 + 0.5
        img= torch.permute(DataI, (1,2, 0))
        org_data.append(idx)
        prob_lst.append(class_prob.item())
        psnr_lst.append(psnr_val)
        ssim_lst.append(ssim_val)
        fid_lst.append(fid_val)
    
    attack_acc = attack_correct/float(total)
    #print(f" Attack Performance = {attack_correct} / {total} = {attack_acc}\t")

    return psnr_lst, ssim_lst, fid_lst, org_data, prob_lst

target_epochs=25
clf_loss_train_tr, clf_loss_test_tr=[],[]
for t in tqdm(range(target_epochs)):
    print(f'Epoch {t+1}\n-------------------------------')
    print("+++++++++CLF Training Starting+++++++++")
    tr_loss, result_train=classifier_train(train_loader2, clf_model, optimiser)
    clf_loss_train_tr.append(tr_loss)


loss_train_tr, loss_test_tr=[],[]
for t in tqdm(range(target_epochs)):
    print(f'Epoch {t+1}\n-------------------------------')
    print("+++++++++Target Training Starting+++++++++")
    tr_loss, result_train=target_train(train_loader, target_model, optimiser)
    loss_train_tr.append(tr_loss)

print("+++++++++Target Test+++++++++")

final_acc=target_utility(test_loader, target_model, batch_size=1)
attack_epochs=50

loss_train, loss_test=[],[]
for t in tqdm(range(attack_epochs)):
    print(f'Epoch {t+1}\n-------------------------------')
    print("+++++++++Training Starting+++++++++")
    tr_loss=attack_train(test_loader, target_model, clf_model, attack_model, optimiser)
    loss_train.append(tr_loss)



print("**********Test Starting************")
psnr_lst, ssim_lst, fid_lst, org_data, max_prob=attack_test(attack_test_loader, target_model, attack_model)
def Average(lst):
    return sum(lst) / len(lst)

list_of_tuples = list(zip(org_data,  max_prob))

df2 =pd.DataFrame(list_of_tuples,
                  columns=['Img', 'Pro'])

df2.to_csv('/dartfs-hpc/rc/home/h/f0048vh/Sparse_guard/stylegan3/file/r1/bgan.csv', index=False)
def seed2vec(G, seed):
  return np.random.RandomState(seed).randn(1, G.z_dim)

def display_image(image):
  plt.axis('off')
  plt.imshow(image)
  plt.show()

def generate_image(G, z, truncation_psi):
    # Render images for dlatents initialized from random seeds.
    Gs_kwargs = {
        'output_transform': dict(func=tflib.convert_images_to_uint8,
         nchw_to_nhwc=True),
        'randomize_noise': False
    }
    if truncation_psi is not None:
        Gs_kwargs['truncation_psi'] = truncation_psi

    label = np.zeros([1] + G.input_shapes[1][1:])
    # [minibatch, height, width, channel]
    images = G.run(z, label, **G_kwargs)
    return images[0]

def get_label(G, device, class_idx):
  label = torch.zeros([1, G.c_dim], device=device)
  if G.c_dim != 0:
      if class_idx is None:
          ctx.fail("Must specify class label with --class when using "\
            "a conditional network")
      label[:, class_idx] = 1
  else:
      if class_idx is not None:
          print ("warn: --class=lbl ignored when running on "\
            "an unconditional network")
  return label

def generate_image(device, G, z, truncation_psi, noise_mode='const',
                   class_idx=None):
  z = torch.from_numpy(z).to(device)
  label = get_label(G, device, class_idx)
  img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
  img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(\
      torch.uint8)
  return PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
URL = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/"\
  "versions/1/files/stylegan3-r-ffhq-1024x1024.pkl"

print(f'Loading networks from "{URL}"...')
device = torch.device('cuda')
with dnnlib.util.open_url(URL) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

preprocess = torchvision.transforms.Compose([
                                 #transforms.ToPILImage(),

                               torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Resize((32)),
                                                          transforms.CenterCrop(32),
                                                         transforms.Normalize(mean=[0.5, 0.5, 0.5],
                          std=[0.5, 0.5, 0.5])
                              ])
csv_file='/dartfs-hpc/rc/home/h/f0048vh/Sparse_guard/stylegan3/file/r1/bgan.csv'
df=pd.read_csv(csv_file)
df['Img'] = df['Img'].str.replace(r'(', '', regex=True).astype('str')
df['Img'] = df['Img'].str.replace(r')', '', regex=True).astype('str')
df['Img'] = df['Img'].str.replace(r',', '', regex=True).astype('str')
df['seed_from'] = df['Img'].str.replace(r'.jpg', '', regex=True).astype('str')

def compare(org_data, recon_img):
  recreated_data = preprocess(recon_img).unsqueeze(0)
  org_data = org_data.to(device)
  recreated_data = recreated_data.to(device)

  psnr = PeakSignalNoiseRatio().to(device)
  psnr_val=abs(psnr(org_data, recreated_data).item())
  #if(psnr_val=='-inf'):
      #psnr_val=Average(psnr_lst)
  #print("PSNR is:", psnr_val)
  ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
  ssim_val=abs(ssim(org_data, recreated_data).item())
  #print("SSIM is:", ssim_val)

  ## Inception Score

  data_scaled=torch.mul(torch.div(torch.sub(org_data, torch.min(org_data)),torch.sub(torch.max(org_data),torch.min(org_data))), 255)
  int_data=data_scaled.to(torch.uint8)
  recon_scaled=torch.mul(torch.div(torch.sub(recreated_data, torch.min(recreated_data)),torch.sub(torch.max(recreated_data),torch.min(recreated_data))), 255)
  int_recon=recon_scaled.to(torch.uint8)
  #print(int_data, int_recon)
  fid_val = abs(calculate_fid(int_data[0][0].cpu().detach().numpy(), int_recon[0][0].cpu().detach().numpy()))
  if (fid_val=='nan'):
      fid_val=Average(fid_lst)
  print('FID is: %.3f' % fid_val)
  return psnr_val, ssim_val, fid_val, recreated_data

 psnr_lst, fid_lst, ssim_lst, run_lst=[], [], [], []
for i in tqdm(range(df.shape[0])):
  run=1
  org_img=df.loc[i, 'Img']
  new_str = org_img.replace("'", "")
  for file in os.listdir('/content/img_align_celeba'):
    if fnmatch.fnmatch(file, new_str):
        print("Yes found!", new_str)
        fp = open("/content/img_align_celeba/"+ new_str,"rb")
        #img = PIL.Image.open(fp)
        org_image = PIL.Image.open(fp)
  org_data = preprocess(org_image).unsqueeze(0)
  sedd_from=df.loc[i, 'seed_from']
  new_seed = sedd_from.replace("'", "")

  def remove_leading_zeros(num):
      return num.lstrip('0')

  SEED_FROM = remove_leading_zeros(new_seed)
  SEED_FROM=int(SEED_FROM)
  SEED_TO = SEED_FROM+1

  # Generate the images for the seeds.
  for k in range(SEED_FROM, SEED_TO):
    print(f"Seed {k}")
    z = seed2vec(G, k)
    recon_img1 = generate_image(device, G, z, truncation_psi=df.loc[i, 'Pro']
  )

  SEED_FROM = rand.randint(0, 1000)
  SEED_TO = SEED_FROM+1

  # Generate the images for the seeds.
  for m in range(SEED_FROM, SEED_TO):
    print(f"Seed {m}")
    z = seed2vec(G, m)
    recon_img2 = generate_image(device, G, z, truncation_psi=df.loc[i, 'Pro']
  )
    
  psnr_val1, ssim_val1, fid_val1, recreated_data1=compare(org_data, recon_img1)
  psnr_val2, ssim_val2, fid_val2, recreated_data2=compare(org_data, recon_img2)
  #org_image.save (f'/content/drive/My Drive/ontest/nvdia/plot/nod/org_img'+str(new_seed)+'.jpg')
  if (psnr_val1>psnr_val2):
    #recon_img1.save (f'//dartfs-hpc/rc/home/h/f0048vh/Sparse_guard/stylegan3/file/r1/plot/bgan/recon_img'+str(new_seed)+'.jpg')
    psnr_val=psnr_val1
    ssim_val=ssim_val1
    fid_val=fid_val1
  else:
    #recon_img2.save (f'/dartfs-hpc/rc/home/h/f0048vh/Sparse_guard/stylegan3/file/r1/plot/bgan/recon_img'+str(new_seed)+'.jpg')
    psnr_val=psnr_val2
    ssim_val=ssim_val2
    fid_val=fid_val2
  #im1.save("geeks.jpg")
  psnr_lst.append(psnr_val)
  ssim_lst.append(ssim_val)
  fid_lst.append(fid_val)
  run_lst.append(run)

average_psnr = Average(psnr_lst)
average_ssim = Average(ssim_lst)
average_incep = Average(fid_lst)
print('Mean scores are>> PSNR, SSIM, FID: ', average_psnr, average_ssim, average_incep)

list_of_tuples = list(zip(psnr_lst,  ssim_lst, fid_lst, run_lst))

df_result =pd.DataFrame(list_of_tuples,
                  columns=['PSNR', 'SSIM', 'FID', 'Run'])
df_result.to_csv('/dartfs-hpc/rc/home/h/f0048vh/Sparse_guard/stylegan3/file/r1/bgan.csv', index=False) 
print("--- %s seconds ---" % (time.time() - start_time))
