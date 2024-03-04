# Step 1: Importing PyTorch and Opacus
import torch
import os
import torchvision
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from skimage import io
from opacus import PrivacyEngine
from numpy import iscomplexobj
from numpy import cov
from numpy import trace
from scipy.linalg import sqrtm
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

n_epochs = 3
batch_size_train = 32
batch_size_test = 16

batch_size = 32
batch_size_train = 32
batch_size_test = 16
num_workers=0
batch_size_attack_test=1
# Step 2: Loading MNIST Data
train_loader = torch.utils.data.DataLoader(datasets.MNIST('/dartfs-hpc/rc/home/h/f0048vh/opacus/data', train=True, download=True,
               transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), 
               (0.3081,)),]),), batch_size=64, shuffle=True, num_workers=1, pin_memory=True)

test_loader = torch.utils.data.DataLoader(datasets.MNIST('/dartfs-hpc/rc/home/h/f0048vh/opacus/data', train=False, 
              transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), 
              (0.3081,)),]),), batch_size=1024, shuffle=True, num_workers=1, pin_memory=True)

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
#data_loader = DataLoader(train_set,   batch_size=gan_batch_size, shuffle=True)

attack_test_loader = DataLoader(train_set_other,   batch_size=batch_size_attack_test, shuffle=True)  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use gpu if available

'''
model = torch.nn.Sequential(torch.nn.Conv2d(1, 16, 8, 2, padding=3), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 1),
        torch.nn.Conv2d(16, 32, 4, 2),  torch.nn.ReLU(), torch.nn.MaxPool2d(2, 1), torch.nn.Flatten(), 
        torch.nn.Linear(32 * 4 * 4, 32), torch.nn.ReLU(), torch.nn.Linear(32, 10))


class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x

    def name(self):
        return "SampleConvNet"
    '''
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
            #nn.BatchNorm2d(16) ,                   
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(), 
            nn.Dropout(.09), 
            nn.MaxPool2d(2, 2),
            #nn.BatchNorm2d(32),                    

                         )
    self.second_part = nn.Sequential(
                           nn.Conv2d(32, 64, 5, 1, 2),     
                            nn.ReLU(), 
                            nn.Dropout(.09), 
                            nn.MaxPool2d(2, 2),
                            #nn.BatchNorm2d(64),   
                            nn.Conv2d(64, 128, 5, 1, 2),     
                            nn.ReLU(), 
                            nn.Dropout(.09), 
                            nn.MaxPool2d(2, 2),
                            #nn.BatchNorm2d(128), 
                            nn.Conv2d(128, 256, 5, 1, 2),     
                            nn.ReLU(), 
                            nn.Dropout(.09),  
                            nn.MaxPool2d(2, 2),
                            #nn.BatchNorm2d(256),  
                            
                           #scancel nn.Softmax(dim=-1),
                         )
    self.third_part = nn.Sequential(
                            nn.Linear(256*2*2, 500),
                            nn.ReLU(),
                            nn.Linear(500, 6),
       

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



target_model = SplitNN().to(device=device)
class Attacker(nn.Module):
  def __init__(self):
    super(Attacker, self).__init__()
    self.layers= nn.Sequential(
                     nn.Linear(256, 512),
                      nn.ReLU(),
                      nn.Linear(512, 32),
                      nn.ReLU(),
                      nn.ConvTranspose2d(32, 16, 5, 1, 2, bias=False),
                      #nn.BatchNorm2d(16),
                      nn.ReLU(),
                      nn.ConvTranspose2d(16, 10, 5, 1, 2, bias=False),
                      #nn.BatchNorm2d(10),
                      nn.ReLU(),
                      nn.ConvTranspose2d(10, 3, 5, 1, 2, bias=False),

                    )
 
  def forward(self, x):
    return self.layers(x)
  
attack_model = Attacker().to(device=device)

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

#attack_model = Attacker().to(device=device)

optimizer = torch.optim.SGD(target_model.parameters(), lr=0.05)
# Step 4: Attaching a Differential Privacy Engine to the Optimizer
#privacy_engine = PrivacyEngine(model, sample_size=60000, alphas=range(2,32), 
#                               noise_multiplier=1.3, max_grad_norm=1.0,)
privacy_engine = PrivacyEngine()
model, optimizer_dp, data_loader = privacy_engine.make_private(
    module=target_model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.1,
    max_grad_norm=1.0,
)
#privacy_engine.attach(optimizer)
criterion = torch.nn.CrossEntropyLoss()

# Step 5: Training the private model over multiple epochs
def train(model, train_loader, optimizer_dp, epoch, device, delta):
    model.train()
    losses = []    
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer_dp.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer_dp.step()
        losses.append(loss.item())  
        '''  
    epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta) 
    print(
        f"Train Epoch: {epoch} \t"
        f"Loss: {np.mean(losses):.6f} "
        f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}")
        '''
    epsilon = privacy_engine.accountant.get_epsilon(delta=delta)
    print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(eps = {epsilon:.2f},  delta= {delta})"
        )
def test(model, test_loader, device):
    model.eval()
    #criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)
'''
privacy_engine = PrivacyEngine()
model, optimizer_dp, data_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.1,
    max_grad_norm=1.0,
)

model.train()
losses = []    
for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
    data, target = data.to(device), target.to(device)
    optimizer_dp.zero_grad()
    output = model(data)
    loss = criterion(output, target)
epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
'''


#model.train()
def attack_train(test_loader, model, attack_model, optimizer_dp):
    model= SplitNN().to(device=device)
#for data, targets in enumerate(tqdm(train_loader)):
    for batch, (data, targets) in enumerate(tqdm(test_loader)):
    # Reset gradients
        data, targets = data.to(device=device), targets.to(device=device)
        optimizer_dp.zero_grad()
        #index, data = data   
        #data=data.view(1000, 784)
        #data=torch.transpose(data, 0, 1)
        # First, get outputs from the target model
        #data= data.view(-1,32*32*3)
        target_outputs = model.first_part(data)
        target_outputs = target_outputs.view(1, 32, target_outputs.shape[0], 16*16)
        
        #print(target_outputs.shape)
        #target_outputs= target_outputs.view(-1,32*32*500)
        #target_outputs = model.second_part(target_outputs)        
        # Next, recreate the data with the attacker
        attack_outputs = attack_model(target_outputs)
        #attack_outputs=attack_outputs.unsqueeze(1)
        #print(attack_outputs.shape)
        #print(data.shape)
        attack_outputs=attack_outputs.repeat(16, 1,2, 1)
        #print(attack_outputs.shape)

        #print(data.shape)
        #print(attack_outputs.shape)
        # We want attack outputs to resemble the original data
        loss2 = ((data - attack_outputs)**2).mean()

        # Update the attack model
        loss2.backward()
        optimizer_dp.step()

    return loss2

def attack_test(train_loader, model, attack_model):
    #model= SplitNN().to(device=device)
    psnr_lst, ssim_lst, fid_lst=[], [], []
    correct=0
    attack_correct=0
    total=0
    for batch, (data, targets) in enumerate(tqdm(train_loader)):
        #data = data.view(data.size(0), -1)
        data, targets = data.to(device=device), targets.to(device=device)
        #org_data=data
        #data= data.view(-1,32*32*3)
        target_outputs = model.first_part(data)
        target_outputs = target_outputs.view(1, 32, target_outputs.shape[0], 16*16)
        #target_outputs= target_outputs.view(-1,32*32*500)
        #target_outputs = model.second_part(target_outputs)
        recreated_data = attack_model(target_outputs)
        print(data.shape)
        print(recreated_data.shape)
        recreated_data=recreated_data.repeat(target_outputs.shape[2], 1,int(recreated_data.shape[3]/recreated_data.shape[2]), 1)
        print(recreated_data.shape)

        #print(org_data.shape)
        #print(data.shape)
        #print(recreated_data.shape)
        
        #recreated_data=recreated_data.resize(64, 3, 32, 32)
        #print(recreated_data.shape)
      
        psnr = PeakSignalNoiseRatio().to(device)
        psnr_val=abs(psnr(data, recreated_data).item())
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
        fid_val = abs(calculate_fid(int_data[0][0].cpu().detach().numpy(), int_recon[0][0].cpu().detach().numpy()))
        if (fid_val=='nan'):
           fid_val=Average(fid_lst)
        print('FID is: %.3f' % fid_val)
        #gen_data= recreated_data.view(-1,32*32*3)
        test_output = model(recreated_data)
        #attack_pred = test_output.max(1, keepdim=True)[1] # get the index of the max log-probability
        #print(f"Done with sample: {counter_a}\ton epsilon={epsilon}")

        #if attack_pred.item() == targets.item():
        #    attack_correct += 1        
        _, pred = torch.max(test_output, -1)
        attack_correct += (pred == targets).sum().item()
        total += targets.size(0)
        ## Commented below to skip saving images >> Original and Reconstructed images

        '''
        #DataI = data[0] / 2 + 0.5
        #img= torch.permute(DataI, (1,2, 0))
        #img=img.to(torch.float32)
        #print(img.shape)
        plt.imshow(data[0][0].cpu().detach().numpy(), cmap='gray')
        plt.xticks([])
        plt.yticks([])

        #plt.imshow(mfcc_spectrogram[0][0,:,:].numpy(), cmap='viridis')
        plt.draw()
        plt.savefig(f'./plot/MNIST/linear/org_img{batch}.jpg', dpi=100, bbox_inches='tight')
        plt.imshow(recreated_data[0][0].cpu().detach().numpy(), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        #plt.imshow(mfcc_spectrogram[0][0,:,:].numpy(), cmap='viridis')
        plt.draw()
        plt.savefig(f'./plot/MNIST/linear/recon_img{batch}.jpg', dpi=100, bbox_inches='tight')
        '''
        psnr_lst.append(psnr_val)
        ssim_lst.append(ssim_val)
        fid_lst.append(fid_val)


    #return psnr_lst, ssim_lst, fid_lst
    attack_acc = attack_correct/float(total)
    print(f" Attack Performance = {attack_correct} / {total} = {attack_acc}\t")

    return psnr_lst, ssim_lst, fid_lst

    
for epoch in range(1, 26):
    train(model, data_loader, optimizer_dp, epoch, device=device, delta=1e-5)

result_acc= test(model, test_loader=test_loader, device=device)

print("Test Accuracy: ", result_acc)

attack_epochs=50
loss_train, loss_test=[],[]
for t in tqdm(range(attack_epochs)):
    print(f'Epoch {t+1}\n-------------------------------')
    print("+++++++++Training Starting+++++++++")
    tr_loss=attack_train(test_loader, model, attack_model, optimizer)
    loss_train.append(tr_loss)

print("**********Test Starting************")
psnr_lst, ssim_lst, fid_lst=attack_test(train_loader, model, attack_model)


print('Done!')


average_psnr = Average(psnr_lst)
average_ssim = Average(ssim_lst)
average_incep = Average(fid_lst)
print('Mean scoers are>> PSNR, SSIM, FID: ', average_psnr, average_ssim, average_incep)

#torch.save(attack_model, '/vast/home/sdibbo/def_ddlc/model_attack/etn/MNIST_20_epoch_CNN_linear_attack.pt')
#torch.save(target_model, '/vast/home/sdibbo/def_ddlc/model_target/etn/MNIST_20_epoch_CNN_linear_target.pt')

df = pd.DataFrame(list(zip(*[psnr_lst,  ssim_lst, fid_lst]))).add_prefix('Col')

#df.to_csv('/vast/home/sdibbo/def_ddlc/result/etn/MNIST_20_epoch_CNN_attack_linear.csv', index=False)
