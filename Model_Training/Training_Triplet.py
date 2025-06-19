# %%

import pickle
import yaml
import os
import timm
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader,Sampler
from torchvision import models
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
from imgaug import augmenters as iaa

os.chdir(os.path.dirname(os.path.dirname(__file__)))
with open('Utils/Global_Params.yaml') as file:
    files = yaml.full_load(file) 

import Utils.Transforms as Transforms

# %% Arguments.

parser = argparse.ArgumentParser(description='Triplet training')
parser.add_argument('data', metavar='DIR',
                    help='savedir')
parser.add_argument('--comp_type', default='same_patch',
                    choices=['same_patch', 'close_patch', 'combined_patch'],
                    help='type of comparison between patches')
parser.add_argument('--fold', default='0',
                    choices = ['0', '1', '2'])
parser.add_argument('--epochs', default=10)
parser.add_argument('batch_size', default=32)
parser.add_argument('backbone', default='vitb',
                    choices=['vitb', 'resnet50'])

# %%

class ImgDataSet(Dataset):
    def __init__(self,patches,labels,
                transform=None,preproc_fn= lambda x:np.float32(x)/255):
        self.patches=patches
        self.numberOfPatches=len(patches)
        self.labels=labels        
        self.transform=transform
        self.preproc=preproc_fn

    def __len__(self):
        return self.numberOfPatches
    
    def trans(self, img):
        
        f1 = iaa.Sometimes(0.5, iaa.color.RemoveSaturation(mul=[0.0,0.5]))

        f2 = Transforms.Compose([Transforms.ToPILImage(),
                                 Transforms.RandomApply([Transforms.HEDJitter(theta=0.015),
                                                         Transforms.ColorJitter(0.2, 0.4, 0.2, 0.0)], p=0.8),
                                 Transforms.RandomApply([Transforms.RandomGaussBlur(radius=[0.5, 1.5]),
                                                         Transforms.RandomElastic(alpha=2, sigma=0.06)], p=0.2),
                                 Transforms.AutoRandomRotation(),
                                 Transforms.ToTensor()])
                                
        return f2(f1.augment_image(img))

    def __getitem__(self,idx):
        img=self.patches[idx]
        label=self.labels[idx]
        img = self.trans(img)
        return {'image':img,'label':label}

class PairSampler(Sampler):
    
    def __init__(self,dataset,batchSize,compType):
        assert batchSize%2==0
        self.numBatches=int(np.floor(len(dataset) / batchSize))
        self.compType=compType
        
        self.labels=dataset.labels
        self.batchSize=batchSize
        self.uniqueLabels=np.unique(self.labels)

        self.sampleIdx={}

        for label in self.uniqueLabels:
              self.sampleIdx[label]=np.where(self.labels==label)[0]

    def __iter__(self):

        self.indexList=[]
        for i in range(self.numBatches):
            
            batchIndices=[]
            for inBatchCounter in range(int(self.batchSize/2)):
                sampleNumber=int(np.random.choice(self.uniqueLabels))
                if self.compType == 'close_patch':
                    batchIndices+=np.random.choice(self.sampleIdx[sampleNumber],size=2).tolist()
                elif self.compType == 'same_patch':
                    batchIndices+=np.repeat(np.random.choice(self.sampleIdx[sampleNumber],size=1),2).tolist()
            self.indexList.append(batchIndices)
        return iter(self.indexList)
    
    def __len__(self):
        return self.numBatches  

# %%

if __name__ == "__main__":
    
    args = parser.parse_args()
    fold = args.fold
    foldsPath=files['DATA']['tma']['bigtma_folds']
    foldsFile=os.path.join(foldsPath,'Training_Fold'+str(fold)+'.pkl')
    (trainingPatches,trainingLabels,trainingPunches,trainingSlides)=pickle.load(open(foldsFile,'rb'))
    trainingPunchNames,trainingPunchNumbers=np.unique(trainingLabels,return_inverse=True)

    batchSize=  args.batch_size
    trainDataSet=ImgDataSet(trainingPatches,trainingPunchNumbers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.backbone == 'vitb':
        model = timm.create_model('vit_base_patch16_224',  pretrained=True)
        model.head = nn.Linear(in_features=768, out_features=1000, bias=True)
    elif args.backbone == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(in_features=2048, out_features=1000, bias=True)
    model = model.to(device)

    numEpochs = args.epochs
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
    lossFunction=losses.TripletMarginLoss(distance = CosineSimilarity())

    if args.comp_type == 'combined_patch':
            trainBatchSamplerA=PairSampler(trainDataSet,batchSize,compType='close_patch')
            trainBatchSamplerB=PairSampler(trainDataSet,batchSize,compType='same_patch')
            trainLoaderA=DataLoader(trainDataSet,batch_sampler=trainBatchSamplerA)
            trainLoaderB=DataLoader(trainDataSet,batch_sampler=trainBatchSamplerB)

            for epoch in range(numEpochs):
                print('epoch '+str(epoch))
                for batchCounter, (dataA, dataB) in enumerate(tqdm(zip(trainLoaderA, trainLoaderB))):
                    
                    inputsA = dataA['image'].to(device)
                    labelsA = dataA['label'].to(device) 
                    
                    inputsB = dataB['image'].to(device)
                    labelsB = dataB['label'].to(device) 
                    
                    model.train()
                    optimizer.zero_grad()
                    outputA = model(inputsA)
                    outputB = model(inputsB)
                    
                    lossA = lossFunction(outputA, labelsA)
                    lossB = lossFunction(outputB, labelsB)
                    loss = lossA + lossB
                    
                    print('loss: '+str(loss))
                    
                    loss.backward()
                    optimizer.step()
                    
                torch.save(model.state_dict(), os.path.join(args.data, 'epoch'+str(epoch)+'.pt'))     
            
    else:
        trainBatchSampler=PairSampler(trainDataSet,batchSize, compType = args.comp_type)
        trainLoader=DataLoader(trainDataSet,batch_sampler=trainBatchSampler)

        for epoch in range(numEpochs):
            print('epoch '+str(epoch))
            for batchCounter, data in enumerate(tqdm(trainLoader)):
                
                inputs = data['image'].to(device)
                labels = data['label'].to(device) 
                
                model.train()
                optimizer.zero_grad()
                output = model(inputs)
                
                loss = lossFunction(output, labels)
                print('loss: '+str(loss))
                
                loss.backward()
                optimizer.step()
                
            torch.save(model.state_dict(), os.path.join(args.data, 'epoch'+str(epoch)+'.pt'))


# %%

