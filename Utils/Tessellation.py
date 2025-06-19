# %% 

import torch
import tiler
import numpy as np

from skimage.transform import resize

# %% 

class SlidePatchGenerator():
    """ Uses Keras' generator to create boxes out of the slide. """
    def __init__(self, slide, patchSize = 224, stride = 100, nBx = 16,
                 shuffle=False, downSampleFactor = 1,
                 preproc_fn=lambda x:np.float32(x)/255):
        """ svsDir
        Initialize the generator to create boxes of the slide.

        ### Returns
        - None

        ### Parameters:
        - `slide: slide`  The loaded slide object.
        - `patchSize: int`  Height/width of the output patches
        - `stride: int`  pixels between successive patches.
        - `nBx: int`  batch size will be square of this number

        """

        self.slide = slide
        self.dsf = downSampleFactor
        self.patchSize = patchSize


        self.boxHeight = int(nBx*stride+(patchSize-stride))
        self.boxWidth = int(nBx*stride+(patchSize-stride))
        self.batch_size = int(nBx*nBx)
        self.shuffle = shuffle

        (self.slideWidth,self.slideHeight)=slide.dimensions
        self.rVals, self.cVals= np.meshgrid(np.arange(0,self.slideHeight,self.dsf*nBx*stride),
                                          np.arange(0,self.slideWidth,self.dsf*nBx*stride))
        self.numberOfBoxes=self.rVals.size
        self.rVals.resize(self.rVals.size)
        self.cVals.resize(self.cVals.size)

        self.preproc=preproc_fn
        self.patchify = tiler.Tiler(data_shape=(self.boxHeight,self.boxWidth,3),
                                    tile_shape=(patchSize, patchSize,3),
                                    channel_dimension=2,overlap=patchSize-stride)
        self.current_index=0

    def __iter__(self):
        return self

    def __next__(self):
        """
        Generate one batch of data

        ### Returns
        - `X: np.array()` of shape (numBoxesInBatch, numRows, numCols, numChannels)
        - `Y: List(np.array(), np.array())` where first numpy array is row values in the batch and second numpy array is col value in the batch.

        ### Parameters
        - `index: int`  batchIndex to be called.
        """
        # Generate indexes of the batch
        #worker_info = torch.utils.data.get_worker_info()
        index=self.current_index
        if index<self.numberOfBoxes:
            Y=[0,0]
            assert self.dsf in [1,2], "Slide must be 20X or 40X"

            r=self.rVals[index]
            c=self.cVals[index]

            if self.dsf==1:  # -- Enter this branch if the slide is 20X
                X=np.zeros((self.batch_size,self.patchSize,self.patchSize,3),dtype=np.float32)

                img=np.zeros((self.boxHeight,self.boxWidth,3))

                imgHeight=int(np.minimum(self.boxHeight,self.slideHeight-(r)))
                imgWidth=int(np.minimum(self.boxWidth,self.slideWidth-(c)))

                img[0:imgHeight,0:imgWidth]=np.array(self.slide.read_region((c,r),0,(imgWidth,imgHeight)))[:,:,range(3)]


            else:  #-- Enter this branch if the slide is 40X

                self.magLevel=0

                # same operations as the other branch branch except multiple by dsf
                boxHeightRaw=int(self.dsf*self.boxHeight)
                boxWidthRaw=int(self.dsf*self.boxWidth)

                imgHeight=int(np.minimum(boxHeightRaw,self.slideHeight-(r)))
                imgWidth=int(np.minimum(boxWidthRaw,self.slideWidth-(c)))

                img=np.zeros((boxHeightRaw,boxWidthRaw,3))

                img[0:imgHeight,0:imgWidth]=np.array(self.slide.read_region((c,r),self.magLevel,(imgWidth,imgHeight)))[:,:,range(3)]

                img= resize(img,(self.boxHeight,self.boxWidth))

            patches=[batch for _, batch in self.patchify(img, batch_size=self.batch_size)][0]
            X=self.preproc(patches).transpose(0,3,1,2)
            self.current_index+=1
            return X, Y
        else:
            raise StopIteration

    def __len__(self):
        return self.numberOfBoxes

def Reconstruct_Slide(res,slideGen):

    nPatch=res.shape[0]
    numberOfChannels=res.shape[-1]
    resFlat=np.squeeze(res)
    idxInBatch=np.remainder(np.arange(nPatch),slideGen.batch_size)
    batchNumber=np.uint32(np.floor(np.arange(nPatch)/slideGen.batch_size))
    nBx=int(np.sqrt(slideGen.batch_size))
    # offset based on location within batch
    offC,offR=np.meshgrid(np.arange(nBx),np.arange(nBx))
    offR=offR.flatten()
    offC=offC.flatten()    
   
    
    nBatch=len(slideGen)
    # Location of batch 
    nGridR=len(np.unique(slideGen.rVals))
    nGridC=len(np.unique(slideGen.cVals))
    rVals,cVals=np.meshgrid(np.arange(0,nGridR),np.arange(0,nGridC))  
   
    rVals=rVals.flatten()*nBx
    cVals=cVals.flatten()*nBx   
    
    assert len(rVals)==nBatch
    
    outMat=np.zeros((nGridR*nBx,nGridC*nBx,numberOfChannels))

    
    idR=rVals[batchNumber]+offR[idxInBatch]
    idC=cVals[batchNumber]+offC[idxInBatch]
    
    for c in range(numberOfChannels):
        temp=np.zeros((nGridR*nBx,nGridC*nBx))
        
        temp[idR,idC]=resFlat[:,c]
        outMat[:,:,c]=temp        
    return outMat



# %%

class PatchesInBoxIterator:
    def __init__(self,slide,boxCornerPos, patchSize = 224, stride = 100, nBx = 16,
                downSampleFactor = 1,
                preproc_fn=lambda x:np.float32(x)/255):
        """ svsDir
        Initialize the generator to create boxes of the slide.

        ### Returns
        - None

        ### Parameters:
        - `slide: slide`  The loaded slide object.
        - `patchSize: int`  Height/width of the output patches
        - `stride: int`  pixels between successive patches.
        - `nBx: int`  batch size will be square of this number

        """

        
        dsf = downSampleFactor
      
        #assert dsf in [1,2], "Slide must be 20X or 40X"
        
        boxHeight = int(nBx*stride+(patchSize-stride))
        boxWidth = int(nBx*stride+(patchSize-stride))
        batch_size = int(nBx*nBx)


        (slideWidth,slideHeight)=slide.dimensions
        

        preproc=preproc_fn
        patchify = tiler.Tiler(data_shape=(boxHeight,boxWidth,3),
                                    tile_shape=(patchSize, patchSize,3),
                                    channel_dimension=2,overlap=patchSize-stride)
        #self.current_index=0
        r=boxCornerPos[0]
        c=boxCornerPos[1]

        if dsf==1:  # -- Enter this branch if the slide is 20X
            X=np.zeros((batch_size,patchSize,patchSize,3),dtype=np.float32)

            img=np.zeros((boxHeight,boxWidth,3))

            imgHeight=int(np.minimum(boxHeight,slideHeight-(r)))
            imgWidth=int(np.minimum(boxWidth,slideWidth-(c)))

            img[0:imgHeight,0:imgWidth]=np.array(slide.read_region((c,r),0,(imgWidth,imgHeight)))[:,:,range(3)]


        else:  #-- Enter this branch if the slide is 40X

            magLevel=0

            # same operations as the other branch branch except multiple by dsf
            boxHeightRaw=int(dsf*boxHeight)
            boxWidthRaw=int(dsf*boxWidth)

            imgHeight=int(np.minimum(boxHeightRaw,slideHeight-(r)))
            imgWidth=int(np.minimum(boxWidthRaw,slideWidth-(c)))

            img=np.zeros((boxHeightRaw,boxWidthRaw,3))

            img[0:imgHeight,0:imgWidth]=np.array(slide.read_region((c,r),magLevel,(imgWidth,imgHeight)))[:,:,range(3)]

            img= resize(img,(boxHeight,boxWidth))
                        

        patches=[batch for _, batch in patchify(img, batch_size=batch_size)][0]
        self.X=preproc(patches).transpose(0,3,1,2)
        self.iterator=iter(self.X)
        
    
    
    def __iter__(self):
        return self

    def __next__(self):    
        return next(self.iterator)

class BoxesInImageIterator:
    def __init__(self, slide, boxPositions,patchSize = 224, stride = 100, nBx = 16,
                downSampleFactor = 1,
                preproc_fn=lambda x:np.float32(x)/255):
        
        self.slide=slide
        self.patchSize=patchSize
        self.stride=stride
        self.nBx=nBx
        self.dsf=downSampleFactor
        self.preproc_fn=preproc_fn
        self.boxPositions=boxPositions 
        self.numberOfBoxes=boxPositions.shape[0]

        self.current_df_index = -1

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_df_index == -1:
            if self.current_df_index == self.numberOfBoxes - 1:
                raise StopIteration
            self.current_df_index += 1

            self.current_iterator=PatchesInBoxIterator(self.slide,self.boxPositions[self.current_df_index], 
            patchSize = self.patchSize, stride = self.stride, nBx = self.nBx,
                downSampleFactor = self.dsf,
                preproc_fn=self.preproc_fn)


           
        try:
            result = next(self.current_iterator)
        except StopIteration:
            if self.current_df_index == self.numberOfBoxes - 1:
                raise StopIteration
            else:
                self.current_df_index += 1
                self.current_iterator=PatchesInBoxIterator(self.slide,self.boxPositions[self.current_df_index], 
                    patchSize = self.patchSize, stride = self.stride, nBx = self.nBx,
                    downSampleFactor = self.dsf,
                    preproc_fn=self.preproc_fn)
                
                
                result = next(self.current_iterator)
        
        return result

class TessellationDataset(torch.utils.data.IterableDataset):
    def __init__(self, slide,patchSize = 224, stride = 100, nBx = 16,
            downSampleFactor = 1,
            preproc_fn=lambda x:np.float32(x)/255):
        super(TessellationDataset).__init__()

        self.slide=slide
        self.patchSize=patchSize
        self.stride=stride
        self.nBx=nBx
        self.dsf=downSampleFactor
        self.preproc_fn=preproc_fn
        self.batch_size=nBx*nBx

        (slideWidth,slideHeight)=slide.dimensions
        rVals, cVals= np.meshgrid(np.arange(0,slideHeight,self.dsf*nBx*stride),
                                          np.arange(0,slideWidth,self.dsf*nBx*stride))
        self.numberOfBoxes=rVals.size
        rVals.resize(rVals.size)
        cVals.resize(cVals.size)
        boxPositions=np.transpose(np.vstack((rVals,cVals)))

        self.boxPositions=boxPositions 
        self.numberOfBoxes=boxPositions.shape[0]



    def __len__(self):
        return self.numberOfBoxes*self.batch_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return BoxesInImageIterator(self.slide, self.boxPositions,
            patchSize = self.patchSize, stride = self.stride, nBx = self.nBx,
                downSampleFactor = self.dsf,
                preproc_fn=self.preproc_fn)
        else:
            selectedBoxPositions= np.array([self.boxPositions[ind] for ind in range(self.numberOfBoxes)\
                if (ind % worker_info.num_workers) == worker_info.id])
            
            return BoxesInImageIterator(self.slide, selectedBoxPositions,
            patchSize = self.patchSize, stride = self.stride, nBx = self.nBx,
                downSampleFactor = self.dsf,
                preproc_fn=self.preproc_fn)



def Reconstruct_Slide_Dataset(res,dset):

    nPatch=res.shape[0]
    numberOfChannels=res.shape[-1]
    resFlat=np.squeeze(res)
    idxInBatch=np.remainder(np.arange(nPatch),dset.batch_size)
    batchNumber=np.uint32(np.floor(np.arange(nPatch)/dset.batch_size))
    nBx=dset.nBx
    # offset based on location within batch
    offC,offR=np.meshgrid(np.arange(nBx),np.arange(nBx))
    offR=offR.flatten()
    offC=offC.flatten()    
   
    
    nBatch=int(len(dset)/dset.batch_size)
    # Location of batch 
    nGridR=len(np.unique(dset.boxPositions[:,0]))
    nGridC=len(np.unique(dset.boxPositions[:,1]))
    rVals,cVals=np.meshgrid(np.arange(0,nGridR),np.arange(0,nGridC))  
   
    rVals=rVals.flatten()*nBx
    cVals=cVals.flatten()*nBx   
    
    assert len(rVals)==nBatch
    
    outMat=np.zeros((nGridR*nBx,nGridC*nBx,numberOfChannels))

    
    idR=rVals[batchNumber]+offR[idxInBatch]
    idC=cVals[batchNumber]+offC[idxInBatch]
    
    for c in range(numberOfChannels):
        temp=np.zeros((nGridR*nBx,nGridC*nBx))
        
        temp[idR,idC]=resFlat[:,c]
        outMat[:,:,c]=temp        
    return outMat

def CropTessellation(outMat, slide, dsf, stride):
    downSampleFactor=dsf
    (slideWidth,slideHeight)=slide.dimensions
    outHeight=int(np.ceil((slideHeight-(downSampleFactor*224))/(downSampleFactor*stride)))+1
    outWidth=int(np.ceil((slideWidth-(downSampleFactor*224))/(downSampleFactor*stride)))+1

    outMatCropped=outMat[:outHeight,:outWidth]
    return outMatCropped

