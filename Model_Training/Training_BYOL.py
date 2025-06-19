# https://github.com/lucidrains/byol-pytorch/blob/6717204748c2a4f4f44b991d4c59ce5b99995582/byol_pytorch/byol_pytorch.py

# %%

import yaml 
import os
import sys
import copy
import random
import pickle
import argparse
import timm
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader, Sampler
from imgaug import augmenters as iaa
from tqdm import tqdm
from functools import wraps

os.chdir(os.path.dirname(os.path.dirname(__file__)))
with open('Utils/Global_Params.yaml') as file:
    files = yaml.full_load(file) 

import Utils.Transforms as Transforms

# %%

parser = argparse.ArgumentParser(description='BYOL training')
parser.add_argument('--data', metavar='DIR',
                    help='savedir')
parser.add_argument('--comp_type', default='same_patch',
                    choices=['same_patch', 'close_patch', 'combined_patch'],
                    help='type of comparison between patches')
parser.add_argument('--fold', default='0',
                    choices = ['0', '1', '2', 'all'])
parser.add_argument('--finetune', default=False)
parser.add_argument('--dim_output', default='1000')
parser.add_argument('--epochs', default=10)
parser.add_argument('batch_size', default=32)
parser.add_argument('backbone', default='vitb',
                    choices=['vitb', 'resnet50'])

# %% Helper functions.

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# loss fn:
def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# exponential moving average:
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

def MLP(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

def SimSiamMLP(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        nn.BatchNorm1d(projection_size, affine=False)
    )

class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer = -2, use_simsiam_mlp = False):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.use_simsiam_mlp = use_simsiam_mlp

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        create_mlp_fn = MLP if not self.use_simsiam_mlp else SimSiamMLP
        projector = create_mlp_fn(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_projection = True):
        representation = self.get_representation(x)

        if not return_projection:
            return representation

        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation


# %% Main class.

class BYOL(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer = -2,
        projection_size = 256,
        projection_hidden_size = 4096,
        moving_average_decay = 0.99,
        use_momentum = True):
        
        super().__init__()
        self.net = net
        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer, use_simsiam_mlp=not use_momentum)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # get device of network and make wrapper same device:
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters:
        self.forward(torch.randn(2, 2, 3, image_size, image_size, device=device))
        

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum
        assert self.target_encoder is not None
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(
        self,
        x,
        return_embedding = False,
        return_projection = True
    ):
        assert not (self.training and x.shape[0] == 1)

        if return_embedding:
            return self.online_encoder(x, return_projection = return_projection)

        image_one, image_two = x[:, 0, :, :, :], x[:, 1, :, :, :]

        online_proj_one, _ = self.online_encoder(image_one)
        online_proj_two, _ = self.online_encoder(image_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj_one, _ = target_encoder(image_one)
            target_proj_two, _ = target_encoder(image_two)
            target_proj_one.detach_()
            target_proj_two.detach_()

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean()
    
# %% Dataset, dataloader, and sampler (TMA).

class ImgDataSet(Dataset):
    def __init__(self, patches, labels, compType):
        self.patches = patches
        self.numberOfPatches = len(patches)
        self.labels = labels        
        self.compType = compType
        
    def trans(self, img):
        
        f1 = iaa.Sometimes(0.8, iaa.color.RemoveSaturation(mul=[0.0,0.5]))

        f2 = Transforms.Compose([Transforms.ToPILImage(),
                                 Transforms.RandomApply([Transforms.HEDJitter(theta=0.015),
                                                         Transforms.ColorJitter(0.2, 0.4, 0.2, 0.0)], p=0.8),
                                 Transforms.RandomApply([Transforms.RandomGaussBlur(radius=[0.5, 1.5]),
                                                         Transforms.RandomElastic(alpha=2, sigma=0.06)], p=0.2),
                                 Transforms.AutoRandomRotation(),
                                 Transforms.ToTensor()])
                                
        return f2(f1.augment_image(img))

    def __len__(self):
        return self.numberOfPatches

    def __getitem__(self, idx1):
        
        X = torch.zeros((2, self.patches.shape[3], self.patches.shape[1], self.patches.shape[2]), dtype=torch.float32)

        if self.compType == 'close_patch':
            idx2 = np.random.choice(np.where(self.labels == self.labels[idx1])[0])
        elif self.compType == 'same_patch':
            idx2 = idx1.copy()


        for idxCounter, idx in enumerate([idx1, idx2]):
            currPatch = np.squeeze(self.patches[idx, :, :, :])
            currPatch = self.trans(currPatch)
            X[idxCounter, :, :, :] = currPatch
                
        return X

class PairSampler(Sampler):
    
    def __init__(self,labels,batchSize):
        assert batchSize%2==0
        self.numBatches=int(np.floor(len(labels) / batchSize))
        
        self.labels=labels
        self.batchSize=batchSize
        self.uniqueLabels=np.unique(self.labels)

        self.sampleIdx={}

        for label in self.uniqueLabels:
              self.sampleIdx[label]=np.where(self.labels==label)[0]

    def __iter__(self):
        self.indexList=[]
        for _ in range(self.numBatches):
            self.batchList = []
            self.batchSamples = random.sample(list(self.sampleIdx.keys()), self.batchSize)
            
            for sample in self.batchSamples :
                self.batchList.append(np.random.choice(self.sampleIdx[sample]))
                
            self.indexList.append(self.batchList)
            
        return iter(self.indexList)
    
    def __len__(self):
        return self.numBatches 
   

# %% Run.

if __name__ == "__main__":
    
    args = parser.parse_args()
    device = torch.device("cuda:0")
    
    if args.backbone == 'vitb':
        backbone = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)
        backbone.head = nn.Linear(in_features=768, out_features=int(args.dim_output), bias=True)
    elif args.backbone == 'resnet50':
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device)
        backbone.fc = nn.Linear(in_features=2048, out_features=int(args.dim_output), bias=True)
    
    if args.finetune:
        for name, child in backbone.named_children():
            if name not in ['norm', 'pre_logits', 'head']:
                for param in child.parameters():
                    param.requires_grad = False
                    print('Finetuning')
        
    
    learner = BYOL(backbone,
                   image_size = 224,
                   hidden_layer ='pre_logits')
    
    batchSize = args.batch_size
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, learner.parameters()), lr = 3e-4)

    fold = args.fold
    
    if fold == 'all':
        patchPath = files['DATA']['tma']['bigtma']
        trainingPatches = []
        trainingLabels = []
        trainingPunches = []
        trainingSlides = []
        for f in ['Testing', 'Training', 'Validation']:
            foldsFile=os.path.join(patchPath,str(f)+'.pkl')
            a1, a2, a3, a4 = pickle.load(open(foldsFile,'rb'))
            trainingPatches.append(a1)
            trainingLabels.append(a2)
            trainingPunches.append(a3)
            trainingSlides.append(a4)

        trainingPatches = np.concatenate(trainingPatches)
        trainingLabels = np.concatenate(trainingLabels)
        trainingPunches = np.concatenate(trainingPunches)
        trainingSlides = np.concatenate(trainingSlides)
        trainingPunchNames,trainingPunchNumbers=np.unique(trainingLabels,return_inverse=True)

    else:
        foldsPath=files['DATA']['tma']['bigtma_folds']
        foldsFile=os.path.join(foldsPath,'Training_Fold'+str(fold)+'.pkl')
        (trainingPatches,trainingLabels,trainingPunches,trainingSlides)=pickle.load(open(foldsFile,'rb'))
        trainingPunchNames,trainingPunchNumbers=np.unique(trainingLabels,return_inverse=True)
    
    numEpochs = args.epochs

    if args.comp_type == 'combined_patch':
        trainDataSetA=ImgDataSet(trainingPatches,trainingPunchNumbers, compType='close_patch')
        trainDataSetB=ImgDataSet(trainingPatches,trainingPunchNumbers, compType='same_patch')

        trainBatchSampler=PairSampler(trainingLabels,batchSize)
        trainLoaderA=DataLoader(trainDataSetA,batch_sampler=trainBatchSampler)
        trainLoaderB=DataLoader(trainDataSetB,batch_sampler=trainBatchSampler)
        
        for epoch in range(numEpochs):
            print('epoch '+str(epoch))
            for batchCounter, (dataA, dataB) in enumerate(tqdm(zip(trainLoaderA, trainLoaderB))):
                imagesA = dataA.to(device)
                imagesB = dataB.to(device)
                lossA = learner(imagesA)
                lossB = learner(imagesB) 
                loss = lossA+lossB
                print('lossA: '+str(lossA))
                print('lossB: '+str(lossB))
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                # update moving average of target encoder:
                learner.update_moving_average() 
                
            torch.save(backbone.state_dict(), os.path.join(args.data, 'epoch'+str(epoch)+'.pt'))
        
    else:
        trainDataSet=ImgDataSet(trainingPatches,trainingPunchNumbers, compType=args.comp_type)
        trainBatchSampler=PairSampler(trainingLabels,batchSize)
        trainLoader=DataLoader(trainDataSet,batch_sampler=trainBatchSampler)

        for epoch in range(numEpochs):
            print('epoch '+str(epoch))
            for batchCounter, data in enumerate(tqdm(trainLoader)):
                images = data.to(device)
                loss = learner(images)
                print('loss: '+str(loss))
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                # update moving average of target encoder:
                learner.update_moving_average() 
                
            torch.save(backbone.state_dict(), os.path.join(args.data, 'epoch'+str(epoch)+'.pt'))
