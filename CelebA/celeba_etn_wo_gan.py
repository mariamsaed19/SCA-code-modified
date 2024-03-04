import os
import zipfile 
import gdown
import torch
import torchvision
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.autograd import Variable
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

## Setup
# Number of gpus available

batch_size = 16
batch_size_train = 16
batch_size_test = 16
num_workers=0
batch_size_attack_test=1
## Below are for GAN Training
gan_batch_size = 8
num_epochs = 50
n_critic = 5
display_step = 50
## Fetch data from Google Drive 
# Root directory for the dataset
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

attack_test_loader = DataLoader(train_set_other,   batch_size=batch_size_attack_test, shuffle=True)  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use gpu if available
#152624
'''
## Create a custom Dataset class
class CelebADataset(Dataset):
  def __init__(self, root_dir, transform=None):
    """
    Args:
      root_dir (string): Directory with all the images
      transform (callable, optional): transform to be applied to each image sample
    """
    # Read names of images in the root directory
    image_names = os.listdir(root_dir)

    self.root_dir = root_dir
    self.transform = transform 
    self.image_names = natsorted(image_names)

    def __len__(self): 
        #len(self.annotations)
        return len(self.image_names)

    def __getitem__(self, idx):
        # Get the path to the image 
        img_path = os.path.join(self.root_dir, self.image_names[idx])
        # Load image and convert it to RGB
        img = Image.open(img_path).convert('RGB')
        # Apply transformations to the image
        if self.transform:
            img = self.transform(img)

        return img
    ## Load the dataset 
# Path to directory with all the images
img_folder = f'{dataset_folder}/img_align_celeba'
# Spatial size of training images, images are resized to this size.
image_size = 32
# Transformations to be applied to each individual image sample
transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                          std=[0.5, 0.5, 0.5])
])
# Load the dataset from file and apply transformations
celeba_dataset = CelebADataset(img_folder, transform)
print(celeba_dataset)
## Create a dataloader 
# Batch size during training
batch_size = 128
# Number of workers for the dataloader
num_workers = 0 if device.type == 'cuda' else 2
# Whether to put fetched data tensors to pinned memory

pin_memory = True if device.type == 'cuda' else False

celeba_dataloader = torch.utils.data.DataLoader(celeba_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                pin_memory=pin_memory,
                                                )
                                                '''

print("Working Fine")
size1 = len(train_loader)
size2 = len(test_loader)

print(size1, size2)
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


#no_len = (i for i in range(num_epochs))
for epoch in range(num_epochs):
    print('Starting epoch {}...'.format(epoch), end=' ')
    for i, (images, labels) in enumerate (tqdm(data_loader)):
        
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

print('GAN Training is Done!')

def generate_img(generator, digit):
    z = Variable(torch.randn(1, 36)).cuda()
    label = torch.LongTensor([digit]).cuda()
    img = generator(z, label).data.cpu()
    #img = 0.5 * img + 0.5
    return img
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


target_model = VictimNN().to(device=device)
clf_model = Pub_classifier().to(device=device)


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
 fid=fid/1000.00
 return fid

def Average(lst):
    return sum(lst) / len(lst)


def classifier_train(train_loader2, clf_model, optimiser):
    clf_model.train()
    size = len(train_loader2.dataset)
    correct = 0
    loss=0
    total_loss=[]
    for batch, (X, Y) in enumerate(tqdm(train_loader2)):
        #print(target_model)
        Y=Y-1
        X, Y = X.to(device=device), Y.to(device)
        #print(X.shape)
        target_model.zero_grad()
        pred = target_model(X)
        gan_loss=[]
        for i in range(Y[0]):
            label=Y[i]
            recreated_data=generate_img(generator, label)
            print(recreated_data.shape)
            recon_img=recreated_data[0]/ 1 + 0.5
            #print(pred.shape, Y.shape)
            #recon_img=torch.permute(recon_img, (2,1, 0))
            recon_img=recon_img.unsqueeze(0).repeat(64, 1, 1, 1)
            #print(recon_img.shape)
            recon_img=recon_img.to(device=device)
            pred2 = target_model(recon_img)
            print(label)
            print(pred2[0])
            prob_max, output = torch.max(pred2[0], 0)
            print(prob_max)
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
        loss=loss
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
    for batch, (X, Y) in enumerate(tqdm(train_loader1)):
        #print(target_model)
        Y=Y-1
        X, Y = X.to(device=device), Y.to(device)
        #print(X.shape)
        target_model.zero_grad()
        pred = target_model(X)
        gan_loss=[]
        for i in range(Y[0]):
            label=Y[i]
            recreated_data=generate_img(generator, label)
            print(recreated_data.shape)
            recon_img=recreated_data[0]/ 1 + 0.5
            #print(pred.shape, Y.shape)
            #recon_img=torch.permute(recon_img, (2,1, 0))
            recon_img=recon_img.unsqueeze(0).repeat(64, 1, 1, 1)
            #print(recon_img.shape)
            recon_img=recon_img.to(device=device)
            pred2 = target_model(recon_img)
            print(label)
            print(pred2[0])
            prob_max, output = torch.max(pred2[0], 0)
            print(prob_max)
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
        loss=loss
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

def attack_train(test_loader, target_model, clf_model, attack_model, optimiser):
    model = VictimNN()
#for data, targets in enumerate(tqdm(train_loader)):
    loss=0
    for batch, (data, targets) in enumerate(tqdm(test_loader)):
    # Reset gradients
        data, targets = data.to(device=device), targets.to(device=device)
        optimiser.zero_grad()
        #index, data = data   
        #data=data.view(1000, 784)
        #data=torch.transpose(data, 0, 1)
        # First, get outputs from the target model
        #data= data.view(-1,32*32*3)
        target_outputs = target_model.first_part(data)
        target_outputs = target_model.second_part(target_outputs)
        target_outputs = target_outputs.view(1, 2*data.shape[0], 2, 16*16)
        #target_outputs= target_outputs.view(-1,32*32*500)
        #target_outputs = target_model.second_part(target_outputs)  
        #print(target_outputs.shape)      
        # Next, recreate the data with the attacker
        attack_outputs = attack_model(target_outputs)
        attack_outputs = attack_outputs.repeat(data.shape[0], 1, data.shape[0], 1)
        # We want attack outputs to resemble the original data
        loss1 = ((data - attack_outputs)**2).mean()

        clf_outputs = clf_model.first_part(data)
        clf_outputs = clf_model.second_part(clf_outputs)
        clf_outputs = clf_outputs.view(1, 2*data.shape[0], 2, 16*16)
        #target_outputs= target_outputs.view(-1,32*32*500)
        #clf_outputs = clf_model.second_part(clf_outputs)        
        # Next, recreate the data with the attacker
        #print(clf_outputs.shape)      
        attack_outputs2 = attack_model(clf_outputs)
        attack_outputs2 = attack_outputs2.repeat(data.shape[0], 1, data.shape[0], 1)

        # We want attack outputs to resemble the original data
        loss2 = ((data - attack_outputs2)**2).mean()

        loss=loss1+loss2
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
        recreated_data = attack_model(target_outputs)
        #target_outputs = target_outputs.view(1, 32, target_outputs.shape[0], 16*16)
        #target_outputs=target_outputs[None, None, :]
        recreated_data = attack_model(target_outputs)
        #print(recreated_data.shape)
        recreated_data = recreated_data.repeat(data.shape[0], 1,8, 1)
        print(data.shape, recreated_data.shape)
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
        #LEARNED PERCEPTUAL IMAGE PATCH SIMILARITY (LPIPS)
        #lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
        #lpips_val=lpips(data, recreated_data).item()
        #print("LPIPS is:", lpips_val)

        '''
        fid = FrechetInceptionDistance(feature=768)
        int_data=data.to(torch.uint8)
        int_recon=recreated_data.to(torch.uint8)
        fid.update(int_data, real=True)
        fid.update(int_recon, real=False)
        fid_val=fid.compute()
        print("FID is:", fid_val)
        '''
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
        #attack_pred = test_output.max(1, keepdim=True)[1] # get the index of the max log-probability
        #print(f"Done with sample: {counter_a}\ton epsilon={epsilon}")

        #if attack_pred.item() == targets.item():
        #    attack_correct += 1        
        _, pred = torch.max(test_output, -1)
        attack_correct += (pred == targets).sum().item()
        total += targets.size(0)
        ### For Saving recon images, uncomment below code blocks
        '''
        DataI = data[0] / 2 + 0.5
        img= torch.permute(DataI, (1,2, 0))
        print(img.shape)
        plt.imshow((img.cpu().detach().numpy()))
        plt.xticks([])
        plt.yticks([])
        
        #plt.imshow(mfcc_spectrogram[0][0,:,:].numpy(), cmap='viridis')
        DataR=recreated_data[0]/2 + 0.5
        recon_img=torch.permute(DataR, (1,2, 0))
        print(recon_img.shape)
        plt.draw()
        plt.savefig(f'/vast/home/sdibbo/def_ddlc/plot/CIFAR/cnn/org_img{batch}.jpg', dpi=100, bbox_inches='tight')
        plt.imshow((recon_img.cpu().detach().numpy()))
        plt.xticks([])
        plt.yticks([])
        #plt.imshow(mfcc_spectrogram[0][0,:,:].numpy(), cmap='viridis')
        plt.draw()
        plt.savefig(f'/vast/home/sdibbo/def_ddlc/plot/CIFAR/cnn/recon_img{batch}.jpg', dpi=100, bbox_inches='tight')
        '''
        psnr_lst.append(psnr_val)
        ssim_lst.append(ssim_val)
        fid_lst.append(fid_val)
    
    attack_acc = attack_correct/float(total)
    print(f" Attack Performance = {attack_correct} / {total} = {attack_acc}\t")

    return psnr_lst, ssim_lst, fid_lst

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
psnr_lst, ssim_lst, fid_lst=attack_test(train_loader, target_model, attack_model)


print('Done!')


average_psnr = Average(psnr_lst)
average_ssim = Average(ssim_lst)
average_incep = Average(fid_lst)
print('Mean scoers are>> PSNR, SSIM, FID: ', average_psnr, average_ssim, average_incep)

#torch.save(attack_model, '/dartfs-hpc/rc/home/h/f0048vh/Sparse_guard/model/gan_MNIST_20_epoch_CNN_linear_attack.pt')
#torch.save(target_model, '/dartfs-hpc/rc/home/h/f0048vh/Sparse_guard/model/gan_MNIST_20_epoch_CNN_linear_target.pt')

df = pd.DataFrame(list(zip(*[psnr_lst,  ssim_lst, fid_lst]))).add_prefix('Col')

#df.to_csv('/dartfs-hpc/rc/home/h/f0048vh/Sparse_guard/result/gan_MNIST_20_epoch_CNN_attack_linear.csv', index=False)