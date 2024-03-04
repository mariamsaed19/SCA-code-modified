import torch
import torchvision
from tqdm import tqdm
from torch import nn, optim
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.inception import InceptionScore
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch.nn.functional as F
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm

n_epochs = 3
batch_size_train = 64
batch_size_attack=1
batch_size_test = 1



train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/dartfs-hpc/rc/home/h/f0048vh/Sparse_guard/data', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/dartfs-hpc/rc/home/h/f0048vh/Sparse_guard/data', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)
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
class LaplacianNoise(nn.Module):
    def __init__(self, sigma=0.5, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.distributions.Laplace(0.0, sigma)

    def forward(self, x):
        if self.training and self.sigma != 0:
            #scale = self.sigma * x if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise 
            x = x + self.noise.sample(x.size()).to(device=device)
        return x
'''
class SplitNN(nn.Module):
  def __init__(self):
    super(SplitNN, self).__init__()
    self.first_part = nn.Sequential(
                           nn.Linear(28, 500),
                           nn.ReLU(),
                           LaplacianNoise(),
                         )
    self.second_part = nn.Sequential(
                           LaplacianNoise(),
                           nn.Linear(14000, 500),
                           nn.ReLU(),
                           nn.Linear(500, 10),
                           nn.Softmax(dim=-1),
                         )
def forward(self, x):
    #x=x.view(-1,32*32*3)
    x=self.first_part(x)
    #print(x.shape)
    #x = torch.flatten(x, 1) # flatten all dimensions except batch
    x = x.view(-1, 14000)
    #print(x.shape)
    x=self.second_part(x)
    return x  
'''
class SplitNN(nn.Module):
  def __init__(self):
    super(SplitNN, self).__init__()
    self.first_part = nn.Sequential(
                           nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.Conv2d(16, 28, 5, 1, 2),     
            nn.ReLU(),                      
                           nn.Linear(28, 500),
                           nn.ReLU(),
                           LaplacianNoise(),
                         )
    self.second_part = nn.Sequential(
                         nn.Linear(392000, 500),
                           nn.ReLU(),
                           nn.Linear(500, 10),
                         )

  def forward(self, x):
    x=self.first_part(x)
    #print(x.shape)
    #x = torch.flatten(x, 1) # flatten all dimensions except batch
    x = x.view(-1, 392000)
    #print(x.shape)
    x=self.second_part(x)
    return x


target_model = SplitNN().to(device=device)

class Attacker(nn.Module):
  def __init__(self):
    super(Attacker, self).__init__()
    self.layers= nn.Sequential(
                     nn.Linear(500, 800),
                      nn.ReLU(),
                      nn.Linear(800, 28),
                      nn.ReLU(),
                      nn.ConvTranspose2d(28, 16, 5, 1, 2, bias=False),
                      nn.BatchNorm2d(16),
                      nn.ReLU(),
                      nn.ConvTranspose2d(16, 16, 5, 1, 2, bias=False),
                      nn.BatchNorm2d(16),
                      nn.ReLU(),
                      nn.ConvTranspose2d(16, 1, 5, 1, 2, bias=False),)
 
  def forward(self, x):
    return self.layers(x)
'''
class Attacker(nn.Module):
  def __init__(self):
    super(Attacker, self).__init__()
    self.layers= nn.Sequential(
                      nn.Linear(512, 800),
                      nn.ReLU(),
                      nn.Linear(800, 32*32*3),
                    )
 
  def forward(self, x):
    return self.layers(x)  
'''    
attack_model = Attacker().to(device=device)
#optimiser = optim.Adam(target_model.parameters(), lr=1e-4)
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

def target_train(train_loader, target_model, optimiser):
    target_model.train()
    size = len(train_loader.dataset)
    correct = 0
    total_loss=[]
    for batch, (X, Y) in enumerate(tqdm(train_loader)):
        #print(target_model)
        X, Y = X.to(device=device), Y.to(device)
        target_model.zero_grad()
        pred = target_model(X)
        print(pred.shape, Y.shape)
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
    model = SplitNN()
#for data, targets in enumerate(tqdm(train_loader)):

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
        # Next, recreate the data with the attacker
        attack_outputs = attack_model(target_outputs)
        
        # We want attack outputs to resemble the original data
        loss = ((data - attack_outputs)**2).mean()

        # Update the attack model
        loss.backward()
        optimiser.step()

    return loss


def target_utility(test_loader, target_model, batch_size=1):
    size = len(test_loader.dataset)
    target_model.eval()
    test_loss, correct = 0, 0
    correct = 0
    counter_a=0
    total=0
    #with torch.no_grad():
    for batch, (X, Y) in enumerate(tqdm(test_loader)):
        X, Y = X.to(device=device), Y.to(device)
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
        init_pred = pred.max(1, keepdim=True)[1] # get the index of the max log-probability

        if init_pred.item() == Y.item():
            correct += 1


    # Calculate final accuracy for this epsilon
    final_acc = correct/float(total)
    print(f"Target Model Accuracy = {correct} / {total} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc 



def attack_test(train_loader, target_model, attack_model):
    model = SplitNN()
    psnr_lst, ssim_lst, fid_lst=[], [], []
    correct=0
    attack_correct=0
    total=0
    for batch, (data, targets) in enumerate(tqdm(train_loader)):
        #data = data.view(data.size(0), -1)
        data, targets = data.to(device=device), targets.to(device=device)
        #org_data=data
        #data= data.view(-1,32*32*3)
        target_outputs = target_model.first_part(data)
        recreated_data = attack_model(target_outputs)
        
        #print(org_data.shape)
        print(data.shape)
        print(target_outputs.shape)
        print(recreated_data.shape)
        recreated_data=recreated_data.resize(targets.size(0), 1, 28, 28)
        #data=torch.squeeze(data)
        #data=torch.squeeze(data, 1)
        #data=data.resize(64, 28)

        print(recreated_data.shape)
      
        psnr = PeakSignalNoiseRatio().to(device)
        psnr_val=psnr(data, recreated_data).item()
        #print("PSNR is:", psnr_val)
        
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        ssim_val=ssim(data, recreated_data).item()
        #print("SSIM is:", ssim_val)

        ## Inception Score
        '''
        data_scaled=torch.mul(torch.div(torch.sub(data, torch.min(data)),torch.sub(torch.max(data),torch.min(data))), 255)
        int_data=data_scaled.to(torch.uint8)
        recon_scaled=torch.mul(torch.div(torch.sub(recreated_data, torch.min(recreated_data)),torch.sub(torch.max(recreated_data),torch.min(recreated_data))), 255)
        int_recon=recon_scaled.to(torch.uint8)
        fid_val = calculate_fid(int_data[0][0].cpu().detach().numpy(), int_recon[0][0].cpu().detach().numpy())
        #print('FID is: %.3f' % fid_val)
        '''
        data_scaled=torch.mul(torch.div(torch.sub(data, torch.min(data)),torch.sub(torch.max(data),torch.min(data))), 255)
        int_data=data_scaled.to(torch.uint8)
        recon_scaled=torch.mul(torch.div(torch.sub(recreated_data, torch.min(recreated_data)),torch.sub(torch.max(recreated_data),torch.min(recreated_data))), 255)
        int_recon=recon_scaled.to(torch.uint8)
        fid_val = calculate_fid(int_data[0][0].cpu().detach().numpy(), int_recon[0][0].cpu().detach().numpy())
        if (fid_val=='nan'):
           fid_val=Average(fid_lst)
        print('FID is: %.3f' % fid_val)
        #print('FID is: %.3f' % fid_val)
        #gen_data= recreated_data.view(-1,32*32*3)
        test_output = target_model(recreated_data)
        #attack_pred = test_output.max(1, keepdim=True)[1] # get the index of the max log-probability
        #print(f"Done with sample: {counter_a}\ton epsilon={epsilon}")

        #if attack_pred.item() == targets.item():
        #    attack_correct += 1        
        _, pred = torch.max(test_output, -1)
        attack_correct += (pred == targets).sum().item()
        total += targets.size(0)
        ## Commenting saving images below
        if (batch!=0):
        #DataI = data[0] / 2 + 0.5
        #img= torch.permute(DataI, (1,2, 0))
        #img=img.to(torch.float32)
        #print(img.shape)
            #plt.imshow(data[0][0].cpu().detach().numpy(), cmap='gray')
            #plt.xticks([])
            #plt.yticks([])

            #plt.imshow(mfcc_spectrogram[0][0,:,:].numpy(), cmap='viridis')
            #plt.draw()
            #plt.savefig(f'/dartfs-hpc/rc/home/h/f0048vh/Sparse_guard/plot/org_img{batch}.jpg', dpi=100, bbox_inches='tight')
            '''
            plt.imshow(recreated_data[0][0].cpu().detach().numpy(), cmap='gray')
            plt.xticks([])
            plt.yticks([])
            #plt.imshow(mfcc_spectrogram[0][0,:,:].numpy(), cmap='viridis')
            plt.draw()
            plt.savefig(f'/dartfs-hpc/rc/home/h/f0048vh/Sparse_guard/plot/1/recon_img{batch}.jpg', dpi=100, bbox_inches='tight')
            '''
        psnr_lst.append(psnr_val)
        ssim_lst.append(ssim_val)
        fid_lst.append(fid_val)


    #return psnr_lst, ssim_lst, fid_lst
    attack_acc = attack_correct/float(total)
    print(f" Attack Performance = {attack_correct} / {total} = {attack_acc}\t")

    return psnr_lst, ssim_lst, fid_lst

target_epochs=25
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
    tr_loss=attack_train(test_loader, target_model, attack_model, optimiser)
    loss_train.append(tr_loss)

print("**********Test Starting************")
psnr_lst, ssim_lst, fid_lst=attack_test(train_loader, target_model, attack_model)


print('Done!')


average_psnr = Average(psnr_lst)
average_ssim = Average(ssim_lst)
average_incep = Average(fid_lst)
print('Mean scoers are>> PSNR, SSIM, FID: ', average_psnr, average_ssim, average_incep)

#torch.save(attack_model, '/dartfs-hpc/rc/home/h/f0048vh/Sparse_guard/model/MNIST_20_epoch_CNN_noisy_linear_attack_.5lambda.pt')
#torch.save(target_model, '/dartfs-hpc/rc/home/h/f0048vh/Sparse_guard/model/MNIST_20_epoch_CNN_noisy_linear_target_.5lambda.pt')

df = pd.DataFrame(list(zip(*[psnr_lst,  ssim_lst, fid_lst]))).add_prefix('Col')

#df.to_csv('/dartfs-hpc/rc/home/h/f0048vh/Sparse_guard/result/MNIST_20_epoch_CNN_attack_noisy_linear_laplacian_.5lambda.csv', index=False)