# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 14:24:53 2022

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torch
import torch.optim as optim
from torch.autograd import Variable
import os
import scipy.io as sio
from datasetTorch import structureData
from datasetLoader import get_images
from PIL import Image 
from torchvision.utils import save_image
from scipy.io import savemat
ngpu = 1
device = "cuda:0" 
size_created = 1215
nz=100
ndf=64
ngf=64
nc=3
EPOCH = 1000
BATCH_SIZE = 64
cc = list(range(1215))
class denoising_model(nn.Module):
  def __init__(self):
    super(denoising_model,self).__init__()
    print('debugg')
    self.encoder=nn.Sequential(
        nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
         nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  
            nn.ReLU(True),
            nn.Conv2d(32, 64, 7, stride=3, padding=1),  
            nn.ReLU(True),
            nn.Conv2d(64, 128, 7, stride=5, padding=1),  
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=5, padding=1) 
       
           
       
           
       )
    
    self.decoder = nn.Sequential( 
           
       nn.ConvTranspose2d(256, 128, 3),  
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 7,stride=3, padding=1,output_padding=1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 24, 7,stride=2, padding=1,output_padding=1),  
            nn.ReLU(True),
            nn.ConvTranspose2d(24, 16, 3, stride=2, padding=1,output_padding=1),  
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1),  
            nn.ReLU(True),
            nn.ConvTranspose2d(8,3, 5, stride=2, padding=1,output_padding=1),  
            nn.Sigmoid()
       
      
        )
 
  def forward(self,x):
    x=self.encoder(x)
    x=self.decoder(x)
    
    return x

created_images = []
names=os.listdir('histogram\denoising')
for b in names[0:size_created]:
   a = sio.loadmat('histogram\denoising/{}'.format(b))
   a = a['inputPatch']
   created_images.append(a)


lim=224

created_images = np.reshape(created_images[0:size_created], (size_created, lim, lim, 3)) 
created_images = np.moveaxis(created_images,3,1)
    
trMeanR = created_images[cc,0,:,:].mean()
trMeanG = created_images[cc,1,:,:].mean()
trMeanB = created_images[cc,2,:,:].mean()
    

    
created_images=torch.from_numpy(created_images)
params = {'batch_size': BATCH_SIZE, 'shuffle': True}   
training_set = created_images[cc].float() 
# training set içerisinde input images ve onların labelları var.(0-1214)
trainingLoader = DataLoader(training_set, **params)  

denoising = denoising_model().to(device)
print(denoising)

optimdenoising = torch.optim.Adam(denoising.parameters(), lr=0.003)
criterion = nn.BCELoss()



def add_noise(img):
    
    noise = torch.rand(img[0].size()) * 0.01
    
    
    noisy_img = img + noise
    return noisy_img


def train(autoencoder, train_loader):
    autoencoder.train()
    avg_loss = 0
    for step, (x) in enumerate(train_loader):
        noisy_x = add_noise(x) 
        noisy_x = noisy_x.view(-1,3,224,224).to(device)
        y = x.view(-1,3,224,224).to(device)

        
        decoded = autoencoder(noisy_x)

        loss = criterion(decoded, y)
        optimdenoising.zero_grad()
        loss.backward()
        optimdenoising.step()
        
        avg_loss += loss.item()
    return avg_loss / len(train_loader)


for epoch in range(1, EPOCH+1):
    loss = train(denoising, trainingLoader)
    ##  HATA ??? RuntimeError: Given groups=1, weight of size [512, 100, 7, 7], 
    ##  expected input[64, 3, 224, 224] to have 100 channels, but got 3 channels 
    ##  instead
    print("[Epoch {}] loss:{}".format(epoch, loss))

testset = torch.utils.data.DataLoader(
    dataset     = created_images,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
    num_workers = 0
)

sample_data = testset.dataset[0].view(-1,3,224,224)
sample_data = sample_data.type(torch.FloatTensor)

original_x = sample_data
noisy_x = add_noise(original_x).to(device)
recovered_x = denoising(noisy_x)



noisy_x = noisy_x.cpu().numpy().transpose(0, 2, 3, 1)
np.stack(noisy_x).astype(None)
mdic = {"inputPatch":noisy_x}
savemat('noisy.mat',mdic)
      
original_x = original_x.numpy().transpose(0, 2, 3, 1)
np.stack(original_x).astype(None)
mdic = {"inputPatch":original_x}
savemat('original.mat',mdic)    

recovered_x = recovered_x.cpu().detach().numpy().transpose(0, 2, 3, 1)
np.stack(recovered_x).astype(None)
mdic = {"inputPatch":recovered_x}
savemat('recovered.mat',mdic) 

    # tensor = tensor*255
    # tensor = np.array(tensor, dtype=np.uint8)
    # if np.ndim(tensor)>3:
    #     assert tensor.shape[0] == 1
    #     tensor = tensor[0]
    # return PIL.Image.fromarray(tensor)
    
    
    
    
#     arr = np.ndarray((1,80,80,1))#This is your tensor
# arr_ = np.squeeze(arr) # you can give axis attribute if you wanna squeeze in specific dimension
# plt.imshow(arr_)
# plt.show()
# png_file = original_x[0].numpy()

#     #tekrar 0-255 aral
# png_file /= png_file.max()/255.0 
# im = Image.fromarray((png_file ).astype(np.uint8)).convert('RGB').show()
# im = im.save(r'D:\Users\Bitirme projesi\08.03 -Deneme\denoising\original.png')



# model = autoencoder().cuda()
# summary(model, (3, 224, 224))

# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)

# total_loss = 0
# for epoch in range(num_epochs):
#     for data in data_loader:
#         # print(data)
#         img = data
#         print("Min Value of input Image = ",torch.min(img))
#         print("Max Value of input Image = ",torch.max(img))        
#         img = Variable(img).cuda()
#         # ===================forward=====================
#         output = model(img)
#         print("Input Image shape = ",img.shape)
#         print("Output Image shape = ",output.shape)
#         loss = criterion(output, img)
#         # ===================backward====================
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     # ===================log========================
#     total_loss += loss.data
#     print('epoch [{}/{}], loss:{:.4f}'
#           .format(epoch+1, num_epochs, total_loss))
#     if epoch % 10 == 0:
#         pic = to_img(output.cpu().data)
#         save_image(pic, './dc_img/image_{}.png'.format(epoch))

# torch.save(model.state_dict(), './conv_autoencoder.pth')
