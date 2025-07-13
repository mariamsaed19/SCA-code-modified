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
from lcapt.lca import LCAConv2D
import os 

n_epochs = 3
batch_size_train = 16
batch_size_test = 16

save_dir = "./result/cifar-cnn/lca2/"
save_dir_imgs = "./result/cifar-cnn/lca2/images"
exp_name = "cifar-cnn_lca2_linear"
os.makedirs(save_dir,exist_ok=True)
os.makedirs(save_dir+"/attack",exist_ok=True)
os.makedirs(save_dir+"/target",exist_ok=True)
os.makedirs(save_dir_imgs,exist_ok=True)


train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR10('./data', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])),
  batch_size=batch_size_train, shuffle=True)
train_loader_aug = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR10('./data', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                                #torchvision.transforms.RandomHorizontalFlip(p=1),
                                #torchvision.transforms.RandomResizedCrop(32, (0.85, 1.0)),
                                torchvision.transforms.Resize((40, 40)),      
                                torchvision.transforms.RandomCrop((32, 32)),  
                                torchvision.transforms.RandomHorizontalFlip(),
                                torchvision.transforms.RandomRotation(15),
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])),
  batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR10('./data', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])),
  batch_size=batch_size_test, shuffle=True)
  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use gpu if available

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
                 lambda_=0.5, lca_iters=500, pad="same"),             
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
                            nn.Linear(256*2*2, 512),
                            nn.ReLU(),
                            nn.Linear(512, 10),
       

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

    return x



target_model = SplitNN().to(device=device ,  dtype=torch.float16)
print(target_model)

# target_model = SplitNN().to(device=device ,  dtype=torch.float16)
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
  
attack_model = Attacker().to(device=device ,  dtype=torch.float16)
optimiser_target = torch.optim.SGD(target_model.parameters(),lr=0.001,momentum=0.9)
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


def target_train(train_loader_aug, target_model, optimiser,epoch):
    target_model.train()
    size = len(train_loader_aug.dataset)
    correct = 0
    total_loss=[]
    for batch, (X, Y) in enumerate(tqdm(train_loader_aug)):
        
        X, Y = X.to(device=device ,  dtype=torch.float16), Y.to(device=device)
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
        data, targets = data.to(device=device ,  dtype=torch.float16), targets.to(device=device)
        optimiser.zero_grad()
        #index, data = data   
        #print(data.shape)
        #data=data.view(1000, 784)
        #data=torch.transpose(data, 0, 1)
        # First, get outputs from the target model
        target_outputs = target_model.first_part(data)
        target_outputs = target_model.second_part(target_outputs)
        # target_outputs = target_model.features(target_outputs)
        # target_outputs = target_model.downsample(target_outputs)
        #print(data.shape)
        print(target_outputs.shape)
        target_outputs = target_outputs.view(1, 2*data.shape[0], 2, 16*16)
        #print(target_outputs.shape)
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
    #target_model.eval()
    test_loss, correct = 0, 0
    correct = 0
    total=0
    counter_a=0
    #with torch.no_grad():
    for batch, (X, Y) in enumerate(tqdm(test_loader)):
        X, Y = X.to(device=device ,  dtype=torch.float16), Y.to(device=device)
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
        correct += (output_res == Y).sum().item()


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
        data, targets = data.to(device=device ,  dtype=torch.float16), targets.to(device=device)
        # print("targets:",targets,targets.shape,targets.dtype)
        target_outputs = target_model.first_part(data)
        target_outputs = target_model.second_part(target_outputs)
        # target_outputs = target_model.features(target_outputs)
        # target_outputs = target_model.downsample(target_outputs)
        # print("after target",target_outputs.shape)
        if (data.shape[0]!=32):
           #data=data.repeat(2, 1, 1, 1)
           target_outputs = target_outputs.view(1,data.shape[0], 4, 16*16)
           target_outputs = target_outputs.repeat(1,2, 1, 1)
        else:
            target_outputs = target_outputs.view(1,data.shape[0], 4, 16*16)
        # print("data:",data.shape)
        # print("before attack",target_outputs.shape)

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
        psnr_val=psnr(data, recreated_data).item()
        #print("PSNR is:", psnr_val)
        
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        ssim_val=ssim(data, recreated_data).item()
        #print("SSIM is:", ssim_val)

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
        #print('FID is: %.3f' % fid_val)
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



print("+++++++++Target Test+++++++++")
target_model = torch.load(f'{save_dir}/target/{exp_name}_target.pt', map_location='cpu',weights_only=False)
attack_model = torch.load(f'{save_dir}/attack/{exp_name}_attack.pt', map_location='cpu',weights_only=False)
# Wrap the models

target_model.to(device)
target_model.eval()
attack_model.to(device)
attack_model.eval()
# final_acc=target_utility(test_loader, target_model, batch_size=1)



print("**********Test Starting************")

psnr_lst, ssim_lst, fid_lst=attack_test(train_loader, target_model, attack_model)
def Average(lst):
    return sum(lst) / len(lst)

print('Done!')


average_psnr = Average(psnr_lst)
average_ssim = Average(ssim_lst)
average_incep = Average(fid_lst)
print('Mean scoers are>> PSNR, SSIM, FID: ', average_psnr, average_ssim, average_incep)


df = pd.DataFrame(list(zip(*[psnr_lst,  ssim_lst, fid_lst]))).add_prefix('Col')

df.to_csv(f'{save_dir}/{exp_name}-result.csv', index=False)