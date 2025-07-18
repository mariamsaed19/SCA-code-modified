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
import os

n_epochs = 3
batch_size_train = 64
batch_size_attack=1
batch_size_test = 1

target_epochs=25
attack_epochs=50
save_dir = "./result/mnist/nod/"
save_dir_imgs = "./result/mnist/nod/images"
exp_name = "MNIST_nod_linear"
os.makedirs(save_dir,exist_ok=True)
os.makedirs(save_dir+"/attack",exist_ok=True)
os.makedirs(save_dir+"/target",exist_ok=True)
os.makedirs(save_dir_imgs,exist_ok=True)

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use gpu if available

class SplitNN(nn.Module):
  def __init__(self):
    super(SplitNN, self).__init__()
    self.first_part = nn.Sequential(
                           nn.Linear(28, 500),
                           nn.ReLU(),
                         )
    self.second_part = nn.Sequential(
                           nn.Linear(500, 500),
                           nn.ReLU(),
                           nn.Linear(500, 28),
                           nn.ReLU(), 
                           nn.Linear(28, 500),                        )
    self.third_part = nn.Sequential(
                           nn.Linear(1*28*500, 10),
                         )

  def forward(self, x):
    x=self.first_part(x)
    x=self.second_part(x)
    x = x.view(-1, 1*28*500)
    x=self.third_part(x)
    return x

target_model = SplitNN().to(device=device)

class Attacker(nn.Module):
  def __init__(self):
    super(Attacker, self).__init__()
    self.layers= nn.Sequential(
                      nn.Linear(500, 800),
                      nn.ReLU(),
                      nn.Linear(800, 28),
                    )
 
  def forward(self, x):
    return self.layers(x)

attack_model = Attacker().to(device=device)
optimiser_target=torch.optim.SGD(target_model.parameters(),lr=0.001,momentum=0.9)
optimiser_attack=torch.optim.SGD(attack_model.parameters(),lr=0.001,momentum=0.9)
cost = torch.nn.CrossEntropyLoss()
def Average(lst):
    return sum(lst) / len(lst)

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


def target_train(train_loader, target_model, optimiser):
    target_model.train()
    size = len(train_loader.dataset)
    correct = 0
    total_loss=[]
    for batch, (X, Y) in enumerate(tqdm(train_loader)):
        X, Y = X.to(device=device), Y.to(device)
        target_model.zero_grad()
        pred = target_model(X)
        loss = cost(pred, Y)
        loss.backward()
        optimiser.step()
        _, output = torch.max(pred, 1)
        correct+= (output == Y).sum().item()
        total_loss.append(loss.item())

    correct /= size
    loss= sum(total_loss)/batch
    result_train=100*correct
    print(f'\nTraining Performance:\nacc: {(100*correct):>0.1f}%, avg loss: {loss:>8f}\n')
    
    return loss, result_train

def attack_train(test_loader, target_model, attack_model, optimiser):
    model = SplitNN()

    for batch, (data, targets) in enumerate(tqdm(test_loader)):
    # Reset gradients
        data, targets = data.to(device=device), targets.to(device=device)
        optimiser.zero_grad()
        target_outputs = target_model.first_part(data)
        target_outputs = target_model.second_part(target_outputs)        
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
    for batch, (X, Y) in enumerate(tqdm(test_loader)):
        X, Y = X.to(device=device), Y.to(device)
        X.requires_grad = True
        pred = target_model(X)
        counter_a=counter_a+1
       
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
    attack_model.eval()
    psnr_lst, ssim_lst, fid_lst=[], [], []
    correct=0
    attack_correct=0
    total=0
    for batch, (data, targets) in enumerate(tqdm(train_loader)):
        data, targets = data.to(device=device), targets.to(device=device)
        target_outputs = target_model.first_part(data)
        target_outputs = target_model.second_part(target_outputs)
        recreated_data = attack_model(target_outputs)

        psnr = PeakSignalNoiseRatio().to(device)
        psnr_val=psnr(data, recreated_data).item()
        
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        ssim_val=ssim(data, recreated_data).item()

        ## Inception Score

        data_scaled=torch.mul(torch.div(torch.sub(data, torch.min(data)),torch.sub(torch.max(data),torch.min(data))), 255)
        int_data=data_scaled.to(torch.uint8)
        recon_scaled=torch.mul(torch.div(torch.sub(recreated_data, torch.min(recreated_data)),torch.sub(torch.max(recreated_data),torch.min(recreated_data))), 255)
        int_recon=recon_scaled.to(torch.uint8)
        fid_val = calculate_fid(int_data[0][0].cpu().detach().numpy(), int_recon[0][0].cpu().detach().numpy())
        if (fid_val=='nan'):
           fid_val=Average(fid_lst)
        print('FID is: %.3f' % fid_val)
        test_output = target_model(recreated_data)
     
        _, pred = torch.max(test_output, -1)
        attack_correct += (pred == targets).sum().item()
        total += targets.size(0)
        ## Commented below to skip saving images >> Original and Reconstructed images

        plt.imshow(data[0][0].cpu().detach().numpy(), cmap='gray')
        plt.xticks([])
        plt.yticks([])

        plt.draw()
        plt.savefig(f'{save_dir_imgs}/org_img{batch}.jpg', dpi=100, bbox_inches='tight')
        plt.imshow(recreated_data[0][0].cpu().detach().numpy(), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.draw()
        plt.savefig(f'{save_dir_imgs}/recon_img{batch}.jpg', dpi=100, bbox_inches='tight')
        psnr_lst.append(psnr_val)
        ssim_lst.append(ssim_val)
        fid_lst.append(fid_val)


    #return psnr_lst, ssim_lst, fid_lst
    attack_acc = attack_correct/float(total)
    print(f" Attack Performance = {attack_correct} / {total} = {attack_acc}\t")

    return psnr_lst, ssim_lst, fid_lst

loss_train_tr, loss_test_tr=[],[]
print("**********Target Starting************")
for t in tqdm(range(target_epochs)):

    tr_loss, result_train=target_train(train_loader, target_model, optimiser_target)
    loss_train_tr.append(tr_loss)

print("+++++++++Target Test+++++++++")

final_acc=target_utility(test_loader, target_model, batch_size=1)


loss_train, loss_test=[],[]
print("**********Attack Starting************")
target_model.eval()
for t in tqdm(range(attack_epochs)):
    tr_loss=attack_train(test_loader, target_model, attack_model, optimiser_attack)
    loss_train.append(tr_loss)

print("**********Test Attack Starting************")
psnr_lst, ssim_lst, fid_lst=attack_test(train_loader, target_model, attack_model)


print('Done!')


average_psnr = Average(psnr_lst)
average_ssim = Average(ssim_lst)
average_incep = Average(fid_lst)
print('Mean scoers are>> PSNR, SSIM, FID: ', average_psnr, average_ssim, average_incep)

torch.save(attack_model, f'{save_dir}/attack/{exp_name}_attack.pt')
torch.save(target_model, f'{save_dir}/target/{exp_name}_target.pt')

df = pd.DataFrame(list(zip(*[psnr_lst,  ssim_lst, fid_lst]))).add_prefix('Col')

df.to_csv(f'{save_dir}/{exp_name}-result.csv', index=False)