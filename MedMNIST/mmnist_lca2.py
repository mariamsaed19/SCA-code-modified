import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from opacus import PrivacyEngine
from numpy import iscomplexobj
from numpy import cov
from numpy import trace
from scipy.linalg import sqrtm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from skimage import io
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from lcapt.lca import LCAConv2D


n_epochs = 3
batch_size_train = 8
batch_size_test = 1


#batch_size_train = 64
#batch_size_attack=1
#batch_size_test = 1


class MedicalMNIST(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.annotations = df
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        
        if self.transform:
            image = self.transform(image)
        
        return (image, y_label)
    

mp = {}
df = []
for idx, category in enumerate(os.listdir("/dartfs-hpc/rc/home/h/f0048vh/Sparse_guard/MedMNIST/data")):
    mp[category] = idx
    print(category)
    for image in os.listdir("/dartfs-hpc/rc/home/h/f0048vh/Sparse_guard/MedMNIST/data/"+category):
        df.append([category+"/"+image, mp[category]])
df = np.array(df)
df = pd.DataFrame(df)
df.head()

dataset = MedicalMNIST(df=df, root_dir="/dartfs-hpc/rc/home/h/f0048vh/Sparse_guard/MedMNIST/data",
                       transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Resize((28)),
                                                         torchvision.transforms.Normalize(
                                 (0.5,), (0.5,))
                              ]))
train_set, test_set = torch.utils.data.random_split(dataset,
                                                   [48954,10000])

train_loader = DataLoader(train_set,   batch_size=batch_size_train, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size_test, shuffle=True)

  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use gpu if available

class SplitNN(nn.Module):
  def __init__(self):
    super(SplitNN, self).__init__()
    self.first_part = nn.Sequential(
       LCAConv2D(out_neurons=16,
                in_neurons=1,                        
                kernel_size=5,              
                stride=1,                   
                 lambda_=0.1, lca_iters=500, pad="same",                
            ), nn.BatchNorm2d(16), 
                        nn.Dropout(.09),                              
       LCAConv2D(out_neurons=28,
                in_neurons=16,                        
                kernel_size=5,              
                stride=1,                   
                 lambda_=0.1, lca_iters=500, pad="same",                
            ),                 nn.BatchNorm2d(28),  
                             nn.Dropout(.09), 

                          nn.Linear(28, 500),
                           nn.ReLU(),
 
                         )
    self.second_part = nn.Sequential(
                           nn.Linear(392000, 500),
                           nn.ReLU(),
                           nn.Linear(500, 6),
                           #nn.Softmax(dim=-1),
                         )

  def forward(self, x):
    x=self.first_part(x)
    #print(x.shape)
    #x = torch.flatten(x, 1) # flatten all dimensions except batch
    x = x.view(-1, 392000)
    #print(x.shape)
    x=self.second_part(x)
    return x

target_model = SplitNN().to(device=device, dtype=torch.float16)
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
                      nn.ConvTranspose2d(16, 1, 5, 1, 2, bias=False),

                    )
 
  def forward(self, x):
    return self.layers(x)
  
attack_model = Attacker().to(device=device, dtype=torch.float16)
optimiser = torch.optim.SGD(target_model.parameters(), lr=0.00001, weight_decay = 0.001, momentum = 0.9) 

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

def target_train(train_loader, target_model, optimiser):
    target_model.train()
    size = len(train_loader.dataset)
    correct = 0
    total_loss=[]
    for batch, (X, Y) in enumerate(tqdm(train_loader)):
        
        X, Y = X.to(device=device,  dtype=torch.float16), Y.to(device)
        target_model.zero_grad()
        pred = target_model(X)
        loss = cost(pred, Y)
        if not torch.isnan(loss):
        #print(pred.shape, Y.shape)
        #print(output.shape, target.shape)
            loss.backward()
            optimiser.step()

        pred=torch.nan_to_num(pred, nan=1.0)
        loss = cost(pred, Y)
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
        data, targets = data.to(device=device, dtype=torch.float16), targets.to(device)
        optimiser.zero_grad()
        #index, data = data   
        #print(data.shape)
        #data=data.view(1000, 784)
        #data=torch.transpose(data, 0, 1)
        # First, get outputs from the target model
        target_outputs = target_model.first_part(data)

        # Next, recreate the data with the attacker
        attack_outputs = attack_model(target_outputs)

        # We want attack outputs to resemble the original data
        loss = ((data - attack_outputs)**2).mean()

        # Update the attack model
        loss.backward()
        optimiser.step()

    return loss

def target_utility(test_loader, target_model, batch_size=64):
    size = len(test_loader.dataset)
    #target_model.train()
    test_loss, correct = 0, 0
    total=0
    correct = 0
    counter_a=0
    with torch.no_grad():
        for batch, (X, Y) in enumerate(tqdm(test_loader)):
            X, Y = X.to(device=device,  dtype=torch.float16), Y.to(device)
            X.requires_grad = False
            pred = target_model(X)
            loss = cost(pred, Y)
            #print(X)
            #if not torch.isnan(loss):        #print(pred.shape, Y.shape)
            #print(output.shape, target.shape)

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
    psnr_lst, ssim_lst, fid_lst=[], [], []
    attack_correct=0
    total=0
    for batch, (data, targets) in enumerate(tqdm(train_loader)):
        data, targets = data.to(device=device, dtype=torch.float16), targets.to(device)
        target_outputs = target_model.first_part(data)
        target_outputs=torch.nan_to_num(target_outputs, nan=1.0)
        target_outputs = target_model.second_part(target_outputs)
        target_outputs=torch.nan_to_num(target_outputs, nan=1.0)
        recreated_data = attack_model(target_outputs)


        psnr = PeakSignalNoiseRatio().to(device)
        psnr_val=psnr(data, recreated_data).item()
        if (psnr_val=='-inf'):
           psnr_val=Average(psnr_lst)
        print("PSNR is:", psnr_val)
        
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        ssim_val=ssim(data, recreated_data).item()
        print("SSIM is:", ssim_val)

        
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
        #attack_pred = test_output.max(1, keepdim=True)[1] # get the index of the max log-probability
        #print(f"Done with sample: {counter_a}\ton epsilon={epsilon}")

        #if attack_pred.item() == targets.item():
        #    attack_correct += 1        
        _, pred = torch.max(test_output, -1)
        attack_correct += (pred == targets).sum().item()
        total += targets.size(0)
        ##Commented if not saving figures
        '''
        #DataI = data[0] / 2 + 0.5
        #print(DataI.shape)
        #img= torch.permute(DataI, (1,2, 0))
        img=data.to(torch.float32)
        plt.imshow(data[0][0].cpu().detach().numpy(), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        
        #plt.imshow(mfcc_spectrogram[0][0,:,:].numpy(), cmap='viridis')
        #DataR=recreated_data[0]/2 + 0.5
        #recon_img=torch.permute(DataR, (1,2, 0))
        recon_img=recreated_data.to(torch.float32)
        #print(recon_img.shape)
        plt.draw()
        plt.savefig(f'/vast/home/sdibbo/def_ddlc/plot/MNIST/lca/org_img{batch}.jpg', dpi=100, bbox_inches='tight')
        
        plt.imshow(recreated_data[0][0].cpu().detach().numpy(), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        #plt.imshow(mfcc_spectrogram[0][0,:,:].numpy(), cmap='viridis')
        plt.draw()
        plt.savefig(f'/dartfs-hpc/rc/home/h/f0048vh/Sparse_guard/plot/lca_split/recon_img{batch}.jpg', dpi=100, bbox_inches='tight')
        '''
        psnr_lst.append(psnr_val)
        ssim_lst.append(ssim_val)
        fid_lst.append(fid_val)

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

final_acc=target_utility(test_loader, target_model, batch_size=64)
attack_epochs=50

loss_train, loss_test=[],[]
for t in tqdm(range(attack_epochs)):
    print(f'Epoch {t+1}\n-------------------------------')
    print("+++++++++Training Starting+++++++++")
    tr_loss=attack_train(test_loader, target_model, attack_model, optimiser)
    loss_train.append(tr_loss)

print("**********Test Starting************")
psnr_lst, ssim_lst, fid_lst=attack_test(train_loader, target_model, attack_model)
def Average(lst):
    return sum(lst) / len(lst)

print('Done!')

average_psnr = Average(psnr_lst)
average_ssim = Average(ssim_lst)
average_incep = Average(fid_lst)
print('Mean scoers are>> PSNR, SSIM, FID: ', average_psnr, average_ssim, average_incep)

#torch.save(attack_model, '/dartfs-hpc/rc/home/h/f0048vh/Sparse_guard/model/MNIST_20_epoch_CNN_1l_lca_0.25_attack.pt')
#torch.save(target_model, '/dartfs-hpc/rc/home/h/f0048vh/Sparse_guard/model/MNIST_20_epoch_CNN_lca_0.25_target.pt')

df = pd.DataFrame(list(zip(*[psnr_lst,  ssim_lst, fid_lst]))).add_prefix('Col')

#df.to_csv('/dartfs-hpc/rc/home/h/f0048vh/Sparse_guard/result/MNIST_20_epoch_CNN_attack_1l_lca_0.25.csv', index=False)