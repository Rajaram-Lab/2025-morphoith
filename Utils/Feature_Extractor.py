# %%

import numpy as np
from scipy.stats import mode
from scipy import ndimage as ndi
from skimage import morphology as morph
from skimage.morphology import skeletonize
from skimage.morphology import medial_axis
from scipy.spatial.distance import pdist,squareform
from skimage import measure
import cv2 as cv2
from abc import ABC,abstractmethod
from multiprocessing import Pool
from math import sqrt,pi as PI
import warnings
import networkx as nx
from math import sqrt, atan2, pi as PI

# %%

def ReadImg(blockCoords):
    (cornerPos,magLevel,blockSizeInt,isResizeNeeded,blockSizeOut,cornerPosOut)=blockCoords
    blockImg=np.array(mySlide.read_region(cornerPos,magLevel,blockSizeInt),order='f')[:,:,range(3)]
    if isResizeNeeded:
        blockOut=cv2.resize(blockImg,blockSizeOut)
    else:
        blockOut=blockImg
    return ((cornerPosOut,blockSizeOut),blockOut)


def ReadSlide(slide,blockWidth=2000,nWorkers=64,downSampleFactor=1,
              normalizer=None,downSampleTolerance=0.025):

    global mySlide
    mySlide=slide

    if np.min(np.abs(np.array(slide.level_downsamples)-downSampleFactor))<\
        downSampleTolerance: 
      
      magLevel=int(np.argmin(np.abs(np.array(slide.level_downsamples)-downSampleFactor)))
      isResizeNeeded=False
      downSampleFactor=slide.level_downsamples[magLevel]

    else:
      magLevel=int(np.where(np.array(slide.level_downsamples)<downSampleFactor)[0][-1])
      isResizeNeeded=True
    
    downSampleExtracted=slide.level_downsamples[magLevel]

    dim=slide.dimensions
    nR=int(np.ceil(dim[1]/downSampleFactor))
    nC=int(np.ceil(dim[0]/downSampleFactor))
    nBR=int(np.ceil(nR/blockWidth))
    nBC=int(np.ceil(nC/blockWidth))
    dSInt=downSampleFactor/downSampleExtracted 


    blockList=[]
    for r in range(nBR):
        for c in range(nBC):
            
            cornerPosOut=np.array([int(c*blockWidth),int(r*blockWidth)])
            cornerPos=np.uint32(cornerPosOut*downSampleFactor)

            rSizeOut=int(min(blockWidth,(nR-cornerPosOut[1])))
            cSizeOut=int(min(blockWidth,(nC-cornerPosOut[0])))
            blockSizeOut=(cSizeOut,rSizeOut)
            rSizeInt=int(dSInt*rSizeOut)
            cSizeInt=int(dSInt*cSizeOut)
            blockSizeInt=(cSizeInt,rSizeInt)
            
            blockList.append((cornerPos,magLevel,blockSizeInt,isResizeNeeded,\
                              blockSizeOut,cornerPosOut))
            
    hImg=np.zeros((nR,nC,3),dtype=np.uint8)

    pool = Pool(processes=nWorkers,maxtasksperchild=100)    
    result=pool.map(ReadImg,blockList)
    for imgStuff in result:
        blockCoords,blockSize=imgStuff[0]
        img=imgStuff[1]
        boxSlice=np.s_[blockCoords[1]:((blockCoords[1]+blockSize[1])),
                       blockCoords[0]:((blockCoords[0]+blockSize[0])),
                       0:3]
        if normalizer is None:
            hImg[boxSlice]=img
        else:
            hImg[boxSlice]=normalizer(img)
    pool.close()
    pool.join()
    return hImg    

def SampleSlide(slide,numberOfPatches=250,patchSize=50,rgbThresh=220):                                                                                                                                                               
                                                                                                                                                                                                           
       lowResImg=np.array(slide.read_region((0,0),
                                            slide.level_count-1,
                                            slide.level_dimensions[slide.level_count-1]))[:,:,range(3)]
       
       lowResPos=np.where(np.any(lowResImg<rgbThresh,axis=-1))
       chosenIdx=np.random.choice(range(lowResPos[0].size),numberOfPatches,replace=True)             
       cX=np.int32(lowResPos[0][chosenIdx]*slide.level_downsamples[slide.level_count-1])                                                                                                                                              
       cY=np.int32(lowResPos[1][chosenIdx]*slide.level_downsamples[slide.level_count-1])
       pixelData=np.zeros((numberOfPatches*patchSize*patchSize,3))                                                         
       for n in range(numberOfPatches): 
           patchImg=np.asarray(slide.read_region((cY[n],cX[n]),0,(patchSize,patchSize)))[:,:,range(3)]
           pixelData[range(n*patchSize*patchSize,(n+1)*patchSize*patchSize),:]=np.resize(patchImg,(patchSize*patchSize,3))
       pixelData=np.uint8(pixelData[np.any(pixelData<rgbThresh,axis=-1),:])   
       isNotGreen=np.logical_and(pixelData[:,1]<1.0*pixelData[:,0],pixelData[:,1]<1.0*pixelData[:,2])
       pixelData=pixelData[isNotGreen,:]
       isNotDark=np.any(pixelData>50,axis=-1)
       pixelData=pixelData[isNotDark,:]
       return pixelData.reshape(pixelData.shape[0],1,3)    
   
# %%
  
def Identity(x):
    return x

class Feature(ABC):
    @abstractmethod
    def __len__(self):
        pass
    @abstractmethod
    def names(self):
        pass
    @abstractmethod
    def profile(self,mask,imgList,objSlice):
        pass

class Location(Feature):
    def __len__(self):
        return 6
    def names(self):
        return ['Y_Start','Y_End','X_Start','X_End','Centroid_Y','Centroid_X']
    def profile(self,mask,img,objSlice):
        yStart=objSlice[0].start
        yEnd=objSlice[0].stop
        xStart=objSlice[1].start
        xEnd=objSlice[1].stop
        centroid=np.mean(np.nonzero(mask),axis=1)
        return np.array([yStart,yEnd,xStart,xEnd,centroid[0]+yStart,centroid[1]+xStart])
    

class Size(Feature):
    def __len__(self):
        return 3
    def names(self):
        return ['Area','BBox_Area','Equivalent_Diameter']
    def profile(self,mask,img,objSlice):
        area= np.sum(mask)
        eqDiameter=sqrt(4 * area / PI)
        return np.array([area,mask.size,eqDiameter])

 
class Percent_Mask_Overlap(Feature):
    def __init__(self,imgNum,maskName):
        self.imgNum=imgNum
        self.maskName=maskName
    def __len__(self):
        return 1
    def names(self):
        return ['Percent_'+self.maskName+'_Overlap']
    def profile(self,mask,imgList,objSlice):
        overlapMask = imgList[self.imgNum]
        mask_and = np.logical_and(mask, overlapMask)
        count = np.sum(mask_and)
        area = np.sum(mask)
        percent = count/area
        return np.array([percent])
    
class Shape(Feature):
    def __len__(self):
        return 3
    def names(self):
        return ['Eccentricity','Major_Axis_Length','Minor_Axis_Length']
    def profile(self,mask,imgList,objSlice):
        l1, l2 = measure._moments.inertia_tensor_eigvals(mask)
        if l1 == 0:
            eccentricity=0
        else :
            eccentricity=sqrt(1-l2/l1)
        majAxLength=4*sqrt(l1)
        minAxLength= 4 * sqrt(l2)
        return np.array([eccentricity,majAxLength,minAxLength])
    
    
class Activation(Feature):
    def __init__(self,imgNum,class_num):
        self.imgNum=imgNum
        self.class_num=class_num
    def __len__(self):
        return 1
    def names(self):
        return ['Avg_' + str(self.class_num) + '_Activation']
    def profile(self,mask,imgList,objSlice):
        activationMask = imgList[self.imgNum][:,:,self.class_num]
        nucleiActivations = np.multiply(mask, activationMask)
        meanActivations = np.sum(nucleiActivations) / np.sum(mask)
        return np.array([meanActivations])
    
# get % of pixels above a certain cutoff for a nuclei
class ActivationBins(Feature):
    def __init__(self,imgNum,class_num):
        self.imgNum=imgNum
        self.class_num=class_num
    def __len__(self):
        return 9
    def names(self):
        return [str(self.class_num) + '_>10', str(self.class_num) + '_>20', str(self.class_num) + '_>30', 
                str(self.class_num) + '_>40', str(self.class_num) + '_>50', str(self.class_num) + '_>60', 
                str(self.class_num) + '_>70', str(self.class_num) + '_>80', str(self.class_num) + '_>90']
    def profile(self,mask,imgList,objSlice):
        activationMask = imgList[self.imgNum][:,:,self.class_num]
        nucleiActivations = np.multiply(mask, activationMask)
        n_pixels = np.sum(mask)
        
        cutoff_10 = np.sum(nucleiActivations > 0.1) / n_pixels
        cutoff_20 = np.sum(nucleiActivations > 0.2) / n_pixels
        cutoff_30 = np.sum(nucleiActivations > 0.3) / n_pixels
        cutoff_40 = np.sum(nucleiActivations > 0.4) / n_pixels
        cutoff_50 = np.sum(nucleiActivations > 0.5) / n_pixels
        cutoff_60 = np.sum(nucleiActivations > 0.6) / n_pixels
        cutoff_70 = np.sum(nucleiActivations > 0.7) / n_pixels
        cutoff_80 = np.sum(nucleiActivations > 0.8) / n_pixels
        cutoff_90 = np.sum(nucleiActivations > 0.9) / n_pixels

        return np.array([cutoff_10, cutoff_20, cutoff_30, cutoff_40, cutoff_50, cutoff_60, cutoff_70, cutoff_80, cutoff_90])
    

class Orientation(Feature):
    def __len__(self):
        return 4
    def names(self):
        return ['yProj','xProj','atan','orientation']
    def profile(self,mask,imgList,objSlice):
        M = measure._moments.moments(np.uint(mask), 3)
        ndim=2
        local_centroid=tuple(M[tuple(np.eye(ndim, dtype=int))] /
                     M[(0,) * ndim])
        mu = measure._moments.moments_central(np.uint8(mask),
                                      local_centroid, order=3)
        inertia_tensor=measure._moments.inertia_tensor(mask, mu)
        a, b, b, c = inertia_tensor.flat
        yProj=-2*b
        xProj=c-a
        warnings.filterwarnings('ignore', '.*divide by zero.*', )
        if xProj ==0 and yProj==0:
            atan=0
        else:
            tan=yProj/xProj
            atan=np.arctan(tan)

        r=sqrt((yProj*yProj)+(xProj*xProj))
        if r>0:
            yProj=yProj/r
            xProj=xProj/r


        if a - c == 0:
            if b < 0:
                orient= -PI / 4.
            else:
                orient= PI / 4.
        else:
            orient= 0.5 * atan2(-2 * b, c - a)     
        return np.array([yProj,xProj,atan,orient])    

class Convexity(Feature):
    def __len__(self):
        return 2
    def names(self):
        return ['Convex_Area','Solidity_Fixed']
    def profile(self,mask,imgList,objSlice):
        convexHull=morph.convex_hull_image(mask)
        convexArea=np.sum(convexHull)
        solidity=np.sum(mask)/convexArea
        return np.array([convexArea,solidity])   
    
def area(tPos):
   return np.abs(tPos[0,0]*(tPos[1,1]-tPos[2,1])+tPos[1,0]*(tPos[2,1]-tPos[0,1])+\
       tPos[2,0]*(tPos[0,1]-tPos[1,1]))/2

def curvature(tPos):
    return 4*area(tPos)/np.prod(pdist(tPos))

def GetCurvatureExtent(mask,minDist=8,maxDist=10,maxRand=50,returnCoords=False):
    cImg=np.NAN*np.ones(mask.shape)
    diam=0
    
    r,c=np.where(mask)
    coords=np.transpose(np.stack([r,c]))
    areNbrs=squareform(pdist(coords))<2
    adj=nx.Graph(areNbrs)
    diam=nx.diameter(adj)
    if np.sum(mask)>=minDist:
        
        path = dict(nx.all_pairs_shortest_path(adj,cutoff=maxDist))
        distMat=np.zeros(areNbrs.shape)
        for i in range(areNbrs.shape[0]):
            for j in range(i):
                if j in path[i]:
                    distMat[i,j]=distMat[j,i]=len(path[i][j])
                else:
                    distMat[i,j]=distMat[j,i]=maxDist+1
                    
                    
        p1,p2=np.where(np.logical_and(distMat>=minDist,distMat<=maxDist))         
        
        if returnCoords:
            cDict={}  
        cList=[]
        if len(p1>0):
            nRand=min(maxRand,len(p1))
            idxList=np.random.choice(len(p1),size=nRand,replace=False)
            for i in range(nRand):
                idx=idxList[i]
                startP=p1[idx]
                endP=p2[idx]
                p=path[startP][endP]
                midP=p[int(len(p) / 2)]
                cv=curvature(coords[[startP,midP,endP],:])
                if returnCoords:
                    if midP in cDict:
                        cDict[midP].append(cv)
                    else:
                        cDict[midP]=[cv]
                cList.append(cv)
                 
            if returnCoords:  
                cImg=np.NAN*np.ones(mask.shape)  
                
                for idx in cDict:
                    cImg[r[idx],c[idx]]=np.mean(cDict[idx])
            outCv=np.mean(cList)
        else:
            outCv=0
    else:
        outCv=0
        
    if returnCoords:
        return outCv,diam,cImg
    else:
        return outCv,diam


class Curvature(Feature):
    def __init__(self,minDist=8,maxDist=10,maxRand=50):
        self.minDist=minDist
        self.maxDist=maxDist
        self.maxRand=maxRand  
        
    def __len__(self):
        return 4
    def names(self):
        return ['Curvature','Extent','MaxDistToEdge','Extent_EdgeDist_Ratio']
    def profile(self,mask,imgList,objSlice):
        skelMask,medDist=medial_axis(mask,return_distance=True)
        maxDistToEdge=np.max(medDist[mask])
        cv,extent=GetCurvatureExtent(skelMask,minDist=self.minDist,\
                                    maxDist=self.maxDist,maxRand=self.maxRand,\
                             returnCoords=False)
        return np.array([cv,extent,maxDistToEdge,extent/maxDistToEdge])    
    
class Skeleton(Feature):
    def __len__(self):
        return 1
    def names(self):
        return ['Skeleton_Length']
    def profile(self,mask,imgList,objSlice):
        skeleton=skeletonize(mask)
        skelLength= np.sum(skeleton)  
        

        return np.array([skelLength])

class Haralick_Texture(Feature):
    def __init__(self,imgNum,minVal,maxVal,featPrefix,transform=Identity):
        self.imgNum=imgNum
        self.featPrefix=featPrefix
        self.transform=transform  
        self.minVal=minVal
        self.maxVal=maxVal

    def __len__(self):
        return 13
    def names(self):
        n=[]
        featNames1=['2nd_moment','contrast','correlation','variance',
                    'inv_diff_moment','sum_avg','sum_variance',
                    'sum_entropy','entropy','diff_var','diff_entropy',
                    'inf_corr1','inf_corr2']    

        for featNum in np.arange(13):
            n.append(self.featPrefix+'_Haralick_'+featNames1[featNum])
                
                
                
        return n
    def profile(self,mask,imgList,objSlice):
        imgVals=(self.transform(imgList[self.imgNum])-self.minVal)/(self.maxVal-self.minVal)
        
        imgVals[imgVals<0]=0
        imgVals[imgVals>1]=1
        imgValsInt=np.uint8(255*imgVals)
        imgValsInt[~mask]=0
        try:
            haralickAll=np.concatenate(mahotas.features.haralick(imgValsInt,ignore_zeros=True))
            haralick=np.mean(haralickAll.reshape((4,13)),axis=0)
        except:
            haralick=np.zeros(13)
        return haralick     


class Label(Feature):
    def __init__(self,imgNum,featPrefix):
        self.imgNum=imgNum
        self.featPrefix=featPrefix
        
    def __len__(self):
        return 1
    def names(self):
        suffixes=['_label']
        return [self.featPrefix+s for s in suffixes]
    def profile(self,mask,imgList,objSlice):
        
        
        return mode(imgList[self.imgNum][mask].flatten())[0][0]

class FeatureExtractor():
    def __init__(self,featureList):
        nFeat=0
        if not all([isinstance(f,Feature) for f in featureList]):
            raise SystemError('Passed classes must be of class Feature')
        self.featureNames=['label']
        for f in featureList:
            if len(f) != len(f.names()) :
               raise SystemError('Wrong Size in '+ f.__name__())
            nFeat=nFeat+len(f)
            self.featureNames=self.featureNames+f.names()
        self.numberOfFeatures=nFeat            
        self.fList=featureList
    
    def Run(self,inputs)   :
        label,mask,img,objSlice=inputs
        outMat=np.zeros(self.numberOfFeatures+1)
        outMat[0]=label
        counter=1
        for i,f in enumerate(self.fList):
            outMat[counter:counter+len(f)]=f.profile(mask,img,objSlice)
            counter+=len(f)
        return outMat    
    
    def Names(self):
        return [f.__name__ for f in self.fList]
     


def ExtractFeatures(mask,featureList,imgList=[],minArea=None,maxArea=None): 
    nObj, labelMat, stats, centroids=cv2.connectedComponentsWithStats(np.uint8(mask),8,cv2.CV_32S)
    if (minArea is not None) and (maxArea is not None):
        mapMat=np.zeros(nObj,dtype=np.int32)
        objCounter=1
        for i in range(1,nObj):
            if stats[i,-1]>=minArea and stats[i,-1]<=maxArea:
                mapMat[i]=objCounter
                objCounter+=1
            else:
                mapMat[i]=0
        labelMat=mapMat[labelMat]
    objects=ndi.measurements.find_objects(labelMat)
    dataList=[]
    for objNum,objSlice in enumerate(objects):
        temp=labelMat[objSlice]
        label=objNum+1
        objMask=temp.copy()==label
        iList=[]
        for i in range(len(imgList)):
            iList.append(imgList[i][objSlice])

        dataList.append((label,objMask,iList,objSlice))

    pool = Pool(processes=64,maxtasksperchild=100)    
    
    featCalc=FeatureExtractor(featureList)  
    result=pool.map(featCalc.Run,dataList)  
    
    pool.close()
    pool.join()
      
    featureMat=np.zeros((len(result),result[0].size))
    for f in result:
        featureMat[int(f[0]-1),:]=f
    featureNames=featCalc.featureNames        
    return featureMat,featureNames

