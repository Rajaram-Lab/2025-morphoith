"""
    Copyright (C) 2025, Rajaram Lab - UTSouthwestern 
    
    This file is part of 2025-morphoith.
    
    2025-morphoith is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    
    2025-morphoith is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License
    along with 2025-morphoith. If not, see <http://www.gnu.org/licenses/>.
    
    Aleksandra W. Nielsen, 2025
"""

# %%

import os
import cv2
import skimage
import yaml
import math
import scipy
import numpy as np
import seaborn as sns
import pandas as pd
import glob as glob
import matplotlib.pyplot as plt
import openslide as oSlide
import matplotlib.colors as mcolors
import matplotlib.patheffects as patheffects

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from concurrent.futures import ThreadPoolExecutor
from scipy import spatial
from skimage import color
from statannot import add_stat_annotation
from matplotlib.lines import Line2D

os.chdir(os.path.dirname(os.path.dirname(__file__)))
with open('Utils/Global_Params.yaml') as file:
    files = yaml.full_load(file) 

import Utils.Feature_Extractor as nfeat
from Utils.Image_Utils import MaskFromXML
from Utils.Tumor_Response import SmoothResponseAndClasses
from Utils.Visualization import apply_plot_settings

# %% Visualization settings, helper functions, useful dictionaries.

slidesToVisualize = pd.read_csv(files['VISUALIZATION']['slides'])
figuresDir = files['FIGURES']
apply_plot_settings()

def image_eosin(img):
    img = color.rgb2hed(img)
    null = np.zeros_like(img[:, :, 0])
    img = np.uint8(255*color.hed2rgb(np.stack((null, img[:, :, 1], null), axis=-1)))
    return img

basePilotDir=files['SLIDES']['pilot']
pilotHneDirs=['Batch0', 'Batch1/HnE/','Batch2/HnE/','Batch3/']
fileToPathDict={}
for d in pilotHneDirs:
    pilotFiles=glob.glob(os.path.join(basePilotDir,d,'*.svs'))
    for f in pilotFiles:
        fileToPathDict[os.path.split(f)[-1].replace('.svs','.txt')]=f

# %% 

def per_patch_calculations(coords, allFeaturesDf, vascMaskOrg, slide, nonNucleiMask, stride=100, windowSize=224):
    """Parallel calculations of per-patch hancrafted features.

    Args:
        coords (list): pairs of coordinates in a slide where the patch is from.
        allFeaturesDf (DataFrame): nuclei features.
        vascMaskOrg (array): binary mask of vasculature.
        slide (openslide): WSI loaded through openslide library.
        nonNucleiMask (array): binary mask of areas without nuclei.
        stride (int, optional): stride used in tessellation. Defaults to 100.
        windowSize (int, optional): patch size. Defaults to 224.

    Returns:
        nonEmptyCoords (array): pairs of coordinates in a slide where the patch has nuclei and vasculature.
        nucleiAreaArray (array): nuclei area values for each coordinate.
        vasculatureArray (array): vascular density values for each coordinate.
        eosinIntensityArray (array): eosin intensity values values for each coordinate.
    """

    nucleiAreaList = []
    vasculatureList = []
    eosinIntensityList = []
    nonEmptyCoordsList = []
    
        
    for coord in coords:
        nucleiFeatures = allFeaturesDf[allFeaturesDf['Centroid_Y'].between(coord[0]*stride,coord[0]*stride+windowSize) 
                                     & allFeaturesDf['Centroid_X'].between(coord[1]*stride,coord[1]*stride+windowSize) ]
        nucleiAreas = nucleiFeatures['Area'].values
        vascPatch = vascMaskOrg[coord[0]*stride:coord[0]*stride+windowSize,coord[1]*stride:coord[1]*stride+windowSize]
        uniqueVasc, uniqueVascCount = np.unique(vascPatch,return_counts=True)
        
        # get median nuclei areas:
        if (nucleiAreas.shape[0] != 0):
            nucleiAreasMedian = np.median(nucleiAreas)   
        else:
            nucleiAreasMedian = 0
        nucleiAreaList.append(nucleiAreasMedian)

        # get eosin intensity:
        hnepatch = np.array(slide.read_region((coord[1]*stride,coord[0]*stride), 0, (windowSize,windowSize)))[:,:,range(3)]
        eosinPatch = image_eosin(hnepatch)
        nonNucleiPatch = nonNucleiMask[coord[0]*stride:coord[0]*stride+windowSize,coord[1]*stride:coord[1]*stride+windowSize]
        nonNucleiPatchIdx1, nonNucleiPatchIdx2 = np.nonzero(nonNucleiPatch)
        
        grayEosinPatch = eosinPatch[:,:,1]
        eosinIntensity = 255 - np.mean(grayEosinPatch[nonNucleiPatchIdx1, nonNucleiPatchIdx2])
        eosinIntensityList.append(eosinIntensity)

        # get vascualture area:
        if (uniqueVasc.shape[0] >= 2):
            vascBg = uniqueVascCount[np.where(uniqueVasc==0)[0]][0]
            vascArea = uniqueVascCount[np.where(uniqueVasc==1)[0]][0]
            vascDensity = vascArea/(vascArea+vascBg)
        else:
            vascDensity = 0
        vasculatureList.append(vascDensity)

        # note which patches have both vasculature and nuclei in them:
        if (nucleiAreas.shape[0] != 0) & (uniqueVasc.shape[0] >= 2):
            nonEmptyCoordsList.append(True)
        else:
            nonEmptyCoordsList.append(False)

    nonEmptyCoords = np.array(nonEmptyCoordsList)
    vasculatureArray = np.array(vasculatureList)
    nucleiAreaArray = np.array(nucleiAreaList)
    eosinIntensityArray = np.array(eosinIntensityList)
    
    return (nonEmptyCoords, nucleiAreaArray, vasculatureArray, eosinIntensityArray)


# %%

def handcrafted_features_analysis(inputList, featuresOfInterest, saveFig, runCal):
    """Performs comparison between MorphoITH features similarity and independently derived nuclear size, vascular density, and eosin intensity. 

    Args:
        inputList (list): list of WSIs to calcualte the handcrafted features for.
        featuresOfInterest (list): nuclei handcrafted features of interest.
        saveFig (bool): allows for saving figures as .svg and .png.
        runCal (bool): allows for either running all calcualtions (True) or focusing on example used for figure generation (False).

    Returns:
        allPairsDf (DataFrame): handcrafted values calculated for each WSI.
    """

    allPairsDf = pd.DataFrame(columns=['Name', 'Top % of absolute differences\nin nuclei area',
                                       'Top % of absolute differences\nin vascular density',
                                       'Top % of absolute differences\nin feature vectors cosine distance',
                                       'Top % of absolute differences\nin eosin intensity'])

    for name in inputList:
        
        inputSamplesList = pd.read_csv(files['DATASETS']['cd31'])['Name'].tolist()
        nameIdx = inputSamplesList.index(int(name))
        
        name = str(name)
        annoFile = os.path.join(files['MASKS']['stardist'],name+'.txt')
        
        if name[:2] == 'KC':
            dataset='pilot'
            svsFile=fileToPathDict[os.path.split(annoFile)[-1]]
        else:
            dataset='cd31'
            svsFile=os.path.join(files['SLIDES']['cd31'],name+'.svs')

        # morphoith:
        profilesFile=os.path.join(files['MASKS']['morphoith'],dataset,name+'.npy')
        profiles=np.load(profilesFile)

        # slide info:
        slide=oSlide.open_slide(svsFile)

        # nuclear mask:
        annoMaskNuclei,_= MaskFromXML(annoFile, [], slideDim=slide.dimensions,downSampleFactor=slide.level_downsamples[0])
        assert annoMaskNuclei.shape[0] == slide.dimensions[1]
        assert annoMaskNuclei.shape[1] == slide.dimensions[0]
        nonNucleiMask = np.logical_not(annoMaskNuclei).astype(int)
        
        # tumor mask:
        tissueMaskFile = os.path.join(files['MASKS']['tissue_classifier'],dataset,name+'.npy')
        tumor = np.load(tissueMaskFile)
        tumor = np.uint8(SmoothResponseAndClasses(tumor)[0])
        tumor[tumor != 5] = 0
        tumor[tumor == 5] = 1
        tumor = np.uint8(skimage.morphology.remove_small_objects(tumor == 1, 100))
        tumor = cv2.erode(np.uint8(tumor),np.ones((1,1),np.uint8))
        
        tumorMag = cv2.resize(np.uint8(tumor),(annoMaskNuclei.shape[1],annoMaskNuclei.shape[0]))
        idx1,idx2 = np.nonzero(tumor)
        coords = np.vstack([idx1,idx2]).T
        profilesFlat = profiles[idx1, idx2, :]
        
        # vasculare mask:
        vascFile = os.path.join(files['MASKS']['vasculature'], dataset, name+'.npz')
        vascMaskOrg = np.array(scipy.sparse.load_npz(vascFile).todense(),dtype=np.uint8)

        # nuclei features:
        annoMaskNucleiTumor = annoMaskNuclei * tumorMag
        allFeatures = nfeat.ExtractFeatures(annoMaskNucleiTumor,featuresOfInterest)
        allFeaturesDf = pd.DataFrame(columns=allFeatures[1],data=allFeatures[0])
        allFeaturesDf = allFeaturesDf[allFeaturesDf['Area'].between(10,300)] 
    
        chunkSize = math.ceil(coords.shape[0] / 100)
        pooledCoords = [coords[i:i + chunkSize] for i in range(0, coords.shape[0], chunkSize)]
        
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda coords: per_patch_calculations(coords, allFeaturesDf, vascMaskOrg, slide, nonNucleiMask), pooledCoords))

        stackedResults = np.hstack(results)
        nonEmptyCoordsBool, nucleiAreaArrayAll, vasculatureArrayAll, eosinIntensityArrayAll = stackedResults
        
        nonEmptyCoordsArray = coords[nonEmptyCoordsBool==True]
        nucleiAreaArray = nucleiAreaArrayAll[nonEmptyCoordsBool==True]
        vasculatureArray = vasculatureArrayAll[nonEmptyCoordsBool==True]
        eosinIntensityArray = eosinIntensityArrayAll[nonEmptyCoordsBool==True]
        nonEmptyProfilesArray = profilesFlat[nonEmptyCoordsBool==True]

        #*#*# VISUALIZATION #*#*#
        slideToVisualize = slidesToVisualize[slidesToVisualize['Reason']=='Handcrafted']['Name'].tolist()
        if name in slideToVisualize:

            # RGB #
            idxR,idxC=np.nonzero(tumor)
            profilesPCA = PCA(n_components=10).fit_transform(profilesFlat)
            m=np.percentile(profilesPCA,[1],axis=0)
            M=np.percentile(profilesPCA,[99],axis=0)
            profilesPcaScaled=np.clip((profilesPCA-m)/(M-m),0,1)
            pcaRGB=np.ones((tumor.shape[0],tumor.shape[1],3))
            pcaRGB[idxR,idxC,:]=profilesPcaScaled[:,range(3)]

            fig = plt.figure(figsize=(5,5))
            plt.imshow(pcaRGB)
            plt.xticks([],[])
            plt.yticks([],[])
            plt.tight_layout()
            plt.title(' MorphoITH')
            fig.patch.set_alpha(0.0)
            if saveFig:
                fig.savefig(os.path.join(figuresDir, 'morphoith_'+name+'.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, 'morphoith_'+name+'.svg'),format='svg',dpi=600,bbox_inches='tight')
            plt.show()

            plotNuclei = np.zeros(shape=(tumor.shape))
            plotNuclei[idx1,idx2] = nucleiAreaArrayAll

            plotVasc = np.zeros(shape=(tumor.shape))
            plotVasc[idx1,idx2] = vasculatureArrayAll

            plotEosin = np.zeros(shape=(tumor.shape))
            plotEosin[idx1,idx2] = eosinIntensityArrayAll

            spotCoords = {'A':[115,95],'B':[55,170],'C':[120,195]}
            spotValues = {'A':[], 'B':[], 'C':[]}
            spotSize = 5
            
            fakePlt = pcaRGB[:300,:470,:]

            fig,ax = plt.subplots(figsize=(5,5))
            for featPlot in [plotNuclei, plotVasc, plotEosin]:
                featPlot = featPlot.astype(float)
                featPlot[tumor==0]=np.nan
                for spot, spotCoord in spotCoords.items():
                    spotValues[spot].append(featPlot[spotCoord[0]:spotCoord[0]+spotSize,spotCoord[1]:spotCoord[1]+spotSize].flatten())
            plt.imshow(fakePlt)
            
            for label, coord in spotCoords.items():

                text_x = coord[1] + spotSize / 2 - 25  
                text_y = coord[0] + spotSize / 2 - 0 
                
                plt.text(text_x, text_y, label, ha='center', va='center', color='white', fontsize=15, fontweight='bold',
                        path_effects=[patheffects.withStroke(linewidth=4, foreground='black', capstyle="round")])

                line1_frame = Line2D([coord[1], coord[1] + spotSize], [coord[0], coord[0] + spotSize], color='black', linewidth=5)
                line2_frame = Line2D([coord[1] + spotSize, coord[1]], [coord[0], coord[0] + spotSize], color='black', linewidth=5)
                
                line1 = Line2D([coord[1], coord[1] + spotSize], [coord[0], coord[0] + spotSize], color='white', linewidth=2)
                line2 = Line2D([coord[1] + spotSize, coord[1]], [coord[0], coord[0] + spotSize], color='white', linewidth=2)
                
                ax.add_artist(line1_frame)
                ax.add_artist(line2_frame)
                ax.add_artist(line1)
                ax.add_artist(line2)
                
            plt.title('MorphoITH\n')
            plt.xticks([])
            plt.yticks([])
            fig.patch.set_alpha(0.0)
            if saveFig:
                fig.savefig(os.path.join(figuresDir, 'morphoith_spots_'+name+'.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, 'morphoith_spots_'+name+'.svg'), format='svg', dpi=600, bbox_inches='tight')
            plt.show()
            
            spotDf = pd.DataFrame(data=spotValues,index=['Nucleus size', 'Vasculature density', 'Eosin intensity'])
            extSpotDf = pd.DataFrame({col: spotDf[col].explode() for col in spotDf.columns})
            extSpotDf_reset = extSpotDf.reset_index().rename(columns={'index': 'Feature'})
            meltedExtSpotDf = extSpotDf_reset.melt(id_vars=['Feature'], var_name='Spot', value_name='Value')
            meltedExtSpotDf = meltedExtSpotDf.dropna()
            meltedExtSpotDf['Value'] = pd.to_numeric(meltedExtSpotDf['Value'], errors='coerce')
            
            spotColors = {'A': 'darkmagenta', 'B':'mediumseagreen', 'C':'royalblue'}
            fig, axes = plt.subplots(1, 3, figsize=(13, 5))
            for i, feat in enumerate(list(meltedExtSpotDf['Feature'].unique())):
                sns.boxplot(ax=axes[i], data=meltedExtSpotDf[meltedExtSpotDf['Feature']==feat], x='Spot', y='Value',hue='Spot',dodge=False,palette=spotColors).legend().remove()
                sns.stripplot(ax=axes[i], data=meltedExtSpotDf[meltedExtSpotDf['Feature']==feat], x='Spot', y='Value',alpha=0.7,s=3,hue='Spot',legend=False,palette=spotColors)
                axes[i].set_title(feat+'\n')
                add_stat_annotation(axes[i], data=meltedExtSpotDf[meltedExtSpotDf['Feature']==feat], x='Spot', y='Value',
                            box_pairs=[('A', 'B'), ('A', 'C'), ('B', 'C')],
                            test='Mann-Whitney', text_format='star', loc='inside', verbose=0, fontsize=13)
                if i != 0:
                    axes[i].set_ylabel('')
                if i != 1:
                    axes[i].set_xlabel('')
            plt.tight_layout()
            fig.patch.set_alpha(0.0)
            if saveFig:
                meltedExtSpotDf.to_csv(os.path.join(figuresDir,  'spots_'+name+'.csv'))
                fig.savefig(os.path.join(figuresDir, 'spots_'+name+'.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir,  'spots_'+name+'.svg'),format='svg',dpi=600,bbox_inches='tight')
            plt.show()
            
            cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ['cornsilk','bisque','orangered','red'])
            cmap.set_under('white', alpha=0)
            maskedZeros = tumor==0

            fig = plt.figure()
            plt.imshow(np.ma.array(plotNuclei, mask=maskedZeros), cmap=cmap, interpolation='none')
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.title('Nucleus size')
            fig.patch.set_alpha(0.0)
            if saveFig:
                fig.savefig(os.path.join(figuresDir, 'nuclei_area_'+name+'.png'),bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, 'nuclei_area_'+name+'.svg'),format='svg',dpi=600,bbox_inches='tight')  
            plt.show()

            cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ['cornsilk','orangered', 'orangered','red'])
            cmap.set_under('white', alpha=0)

            fig = plt.figure()
            plt.imshow(np.ma.array(plotVasc, mask=maskedZeros),cmap=cmap,vmin=0, vmax=1,interpolation='none')
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.title('Vasculature density')
            fig.patch.set_alpha(0.0)
            if saveFig:
                fig.savefig(os.path.join(figuresDir, 'vasculature_density_'+name+'.png'),bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, 'vasculature_density_'+name+'.svg'),format='svg',dpi=600,bbox_inches='tight')  
            plt.show()

            cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ['beige','orangered'])
            cmap.set_under('white', alpha=0)

            fig = plt.figure()
            plt.imshow(np.ma.array(plotEosin, mask=maskedZeros), cmap=cmap, vmin=0, interpolation='none')
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.title('Eosin intensity')
            fig.patch.set_alpha(0.0)
            if saveFig:
                fig.savefig(os.path.join(figuresDir, 'eosin_intensity_'+name+'.png'),bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, 'eosin_intensity_'+name+'.svg'),format='svg',dpi=600,bbox_inches='tight')  
            plt.show()

        # get randomly sampled distribution of features: similarity, nuclei area, vascular density differences:
        numberOfPoints = 1000
        np.random.seed((nameIdx+1)+5)
        chosenPointsIdx = np.random.choice(nonEmptyCoordsArray.shape[0], numberOfPoints)
        np.random.seed()
        
        profilesSimDist = scipy.spatial.distance.pdist(nonEmptyProfilesArray[chosenPointsIdx], 'cosine')
        profilesSimDist = np.sort(profilesSimDist)
        
        chosenFeaturesDiff = {}
        featuresArrays = [("nucleiArea", nucleiAreaArray), 
                          ("vasculature", vasculatureArray), 
                          ("eosinIntensity", eosinIntensityArray)]

        for featureName, featureArray in featuresArrays:

            diff = abs(featureArray[chosenPointsIdx].reshape(-1,1) - featureArray[chosenPointsIdx])
            rows, cols = np.triu_indices(diff.shape[0], k=1)
            diff = diff[rows, cols]
            diff = np.sort(diff)
            chosenFeaturesDiff[featureName] = diff

        if name in slideToVisualize:
            return chosenFeaturesDiff

        # sample pairs depending on features:
        pairsNo = 100
        cutoff = 10
        for pair in range(pairsNo):

            ## IF CHOSEN RANDOMLY ###
            distRandomCoords = 0
            c = 0
            while distRandomCoords < cutoff:
                np.random.seed((nameIdx+1+c)+pair)
                randomCoordsIdx = np.random.choice(nonEmptyCoordsArray.shape[0], 2)
                randomCoords = nonEmptyCoordsArray[randomCoordsIdx]
                distRandomCoords = spatial.distance.euclidean(randomCoords[0], randomCoords[1])
                np.random.seed()
                c+=1
            
            areaA, areaB = nucleiAreaArray[randomCoordsIdx]
            densityA, densityB = vasculatureArray[randomCoordsIdx]
            profileA, profileB = nonEmptyProfilesArray[randomCoordsIdx]
            eosinA, eosinB = eosinIntensityArray[randomCoordsIdx]
            
            coordsInput = randomCoords[0,:]
            profileOfInterest = profiles[coordsInput[0], coordsInput[1],:].reshape(1, -1)

            morphClosestRandomInDist = np.searchsorted(profilesSimDist, cosine_distances(profileA.reshape(1, -1),profileB.reshape(1, -1)))
            morphClosestRandomPerc = (morphClosestRandomInDist / len(profilesSimDist)) * pairsNo
            
            areaClosestRandomInDist = np.searchsorted(chosenFeaturesDiff['nucleiArea'], abs(areaA-areaB))
            areaClosestRandomPerc = (areaClosestRandomInDist / len(chosenFeaturesDiff['nucleiArea'])) * pairsNo
            
            vascClosestRandomInDist = np.searchsorted(chosenFeaturesDiff['vasculature'], abs(densityA-densityB))
            vascClosestRandomPerc = (vascClosestRandomInDist / len(chosenFeaturesDiff['vasculature'])) * pairsNo
            
            eosinClosestRandomInDist = np.searchsorted(chosenFeaturesDiff['eosinIntensity'], abs(eosinA-eosinB))
            eosinClosestRandomPerc = (eosinClosestRandomInDist / len(chosenFeaturesDiff['eosinIntensity'])) * pairsNo
            
            nbrs = NearestNeighbors(n_neighbors=pairsNo+1, algorithm='auto', metric='cosine',n_jobs=16).fit(nonEmptyProfilesArray)
            distances,indices = nbrs.kneighbors(profileOfInterest)


            ### IF CHOSEN BASED ON MORPHOLOGY ###
            for i in range(pairsNo)[1:]:
                idxCheck = indices[0][i]
                coordsCheck = coords[idxCheck,:]
                spatialDist = spatial.distance.euclidean(coordsInput, coordsCheck)
                
                if (coordsInput!=coordsCheck).all() & (spatialDist > cutoff):
                    morphIdx = idxCheck.copy()
                    break

            morphClosestMorph= profilesFlat[morphIdx]
            morphClosestMorphInDist = np.searchsorted(profilesSimDist, cosine_distances(profileA.reshape(1, -1),morphClosestMorph.reshape(1, -1)))
            morphoClosestMorphPerc = (morphClosestMorphInDist / len(profilesSimDist)) * pairsNo
                    
            areaClosestMorph = nucleiAreaArray[morphIdx]
            areaClosestMorphInDist = np.searchsorted(chosenFeaturesDiff['nucleiArea'], abs(areaA-areaClosestMorph))
            areaClosestMorphPerc = (areaClosestMorphInDist / len(chosenFeaturesDiff['nucleiArea'])) * pairsNo
            
            vascClosestMorph = vasculatureArray[morphIdx]
            vascClosestMorphInDist = np.searchsorted(chosenFeaturesDiff['vasculature'], abs(densityA-vascClosestMorph))
            vascClosestMorphPerc = (vascClosestMorphInDist / len(chosenFeaturesDiff['vasculature'])) * pairsNo
                        
            eosinClosestMorph = eosinIntensityArray[morphIdx]
            eosinClosestMorphInDist = np.searchsorted(chosenFeaturesDiff['eosinIntensity'], abs(eosinA-eosinClosestMorph))
            eosinClosestMorphPerc = (eosinClosestMorphInDist / len(chosenFeaturesDiff['eosinIntensity'])) * pairsNo

            ### IF CHOSEN BASED ON NUCLEI AREA ###
            diffNuclei = np.abs(np.array(nucleiAreaArray)-areaA)
            diffNucleiIdx = np.argsort(diffNuclei)
            for i in diffNucleiIdx:
                if (i != randomCoordsIdx[0]) & (i != np.nan):
                    closestNucleiIdx = i
                    break
            
            areaClosestNuclei = nucleiAreaArray[closestNucleiIdx]
            areaClosestNucleiInDist = np.searchsorted(chosenFeaturesDiff['nucleiArea'], abs(areaA-areaClosestNuclei))
            areaClosestNucleiPerc = (areaClosestNucleiInDist / len(chosenFeaturesDiff['nucleiArea'])) * pairsNo
            
            vascClosestNuclei = vasculatureArray[closestNucleiIdx]
            vascClosestNucleiInDist = np.searchsorted(chosenFeaturesDiff['vasculature'], abs(densityA-vascClosestNuclei))
            vascClosestNucleiPerc = (vascClosestNucleiInDist / len(chosenFeaturesDiff['vasculature'])) * pairsNo
            
            morphClosestNuclei = nonEmptyProfilesArray[closestNucleiIdx]
            morphClosestNucleiInDist = np.searchsorted(profilesSimDist, cosine_distances(profileA.reshape(1, -1),morphClosestNuclei.reshape(1, -1)))
            morphClosestNucleiPerc = (morphClosestNucleiInDist / len(profilesSimDist)) * pairsNo
            
            eosinClosestNuclei = eosinIntensityArray[closestNucleiIdx]
            eosinClosestNucleiInDist = np.searchsorted(chosenFeaturesDiff['eosinIntensity'], abs(eosinA-eosinClosestNuclei))
            eosinClosestNucleiPerc = (eosinClosestNucleiInDist / len(chosenFeaturesDiff['eosinIntensity'])) * pairsNo

            ### IF CHOSEN BASED ON VASCULAR DENSITY ###
            diffVasculature = np.abs(vasculatureArray-densityA)
            diffVasculatureIdx = np.argsort(diffVasculature)
            for i in diffVasculatureIdx:
                if (i != randomCoordsIdx[0]) & (i != np.nan):
                    closestVasculatureIdx = i
                    break

            areaClosestVasculature = nucleiAreaArray[closestVasculatureIdx]
            areaClosestVasculatureInDist = np.argmax(chosenFeaturesDiff['nucleiArea'] == abs(areaA-areaClosestVasculature))
            areaClosestVasculaturePerc = (areaClosestVasculatureInDist / len(chosenFeaturesDiff['nucleiArea'])) * pairsNo
            
            vascClosestVasculature = vasculatureArray[closestVasculatureIdx]
            vascClosestVasculatureInDist = np.argmax(chosenFeaturesDiff['vasculature'] == abs(densityA-vascClosestVasculature))
            vascClosestVasculaturePerc = (vascClosestVasculatureInDist / len(chosenFeaturesDiff['vasculature'])) * pairsNo
            
            morphClosestVasculature = nonEmptyProfilesArray[closestVasculatureIdx]
            morphClosestVasculatureInDist = np.searchsorted(profilesSimDist, cosine_distances(profileA.reshape(1, -1),morphClosestVasculature.reshape(1, -1)))
            morphClosestVasculaturePerc = (morphClosestVasculatureInDist / len(profilesSimDist)) * pairsNo
            
            eosinClosestVasculature = eosinIntensityArray[closestVasculatureIdx]
            eosinClosestVasculatureInDist = np.searchsorted(chosenFeaturesDiff['eosinIntensity'], abs(eosinA-eosinClosestVasculature))
            eosinClosestVasculaturePerc = (eosinClosestVasculatureInDist / len(chosenFeaturesDiff['eosinIntensity'])) * pairsNo
            
            
            ### IF CHOSEN BASED ON EOSIN INTENSITY ###
            eosinRandom = eosinIntensityArray[randomCoordsIdx[0]]
            diffEosin = np.abs(eosinIntensityArray-eosinRandom)
            diffEosinIdx = np.argsort(diffEosin)
            for i in diffEosinIdx:
                if (i != randomCoordsIdx[0]) & (i != np.nan):
                    closestEosinIdx = i
                    break

            areaClosestEosin = nucleiAreaArray[closestEosinIdx]
            areaClosestEosinInDist = np.argmax(chosenFeaturesDiff['nucleiArea'] == abs(areaA-areaClosestEosin))
            areaClosestEosinPerc = (areaClosestEosinInDist / len(chosenFeaturesDiff['nucleiArea'])) * pairsNo
            
            vascClosestEosin = vasculatureArray[closestEosinIdx]
            vascClosestEosinInDist = np.argmax(chosenFeaturesDiff['vasculature'] == abs(densityA-vascClosestEosin))
            vascClosestEosinPerc = (vascClosestEosinInDist / len(chosenFeaturesDiff['vasculature'])) * pairsNo
            
            morphClosestEosin = nonEmptyProfilesArray[closestEosinIdx]
            morphClosestEosinInDist = np.searchsorted(profilesSimDist, cosine_distances(profileA.reshape(1, -1),morphClosestEosin.reshape(1, -1)))
            morphClosestEosinPerc = (morphClosestEosinInDist / len(profilesSimDist)) * pairsNo
            
            eosinClosestEosin = eosinIntensityArray[closestEosinIdx]
            eosinClosestEosinInDist = np.searchsorted(chosenFeaturesDiff['eosinIntensity'], abs(eosinA-eosinClosestEosin))
            eosinClosestEosinPerc = (eosinClosestEosinInDist / len(chosenFeaturesDiff['eosinIntensity'])) * pairsNo
            
            pairsDf = pd.DataFrame({'Name':[name]*5,
                                    'Points chosen':['Randomly',
                                                     'Based on\nmorphological similarity',
                                                     'Based on\nnuclei size',
                                                     'Based on\nvasculature density',
                                                     'Based on\neosin intensity'], 
                                    
                                    'Top % of absolute differences\nin nuclei area': [areaClosestRandomPerc,
                                                                                        areaClosestMorphPerc,
                                                                                        areaClosestNucleiPerc,
                                                                                        areaClosestVasculaturePerc,
                                                                                        areaClosestEosinPerc],

                                    'Top % of absolute differences\nin vascular density':[vascClosestRandomPerc,
                                                                                            vascClosestMorphPerc,
                                                                                            vascClosestNucleiPerc,
                                                                                            vascClosestVasculaturePerc,
                                                                                            vascClosestEosinPerc],
                                    
                                    'Top % of absolute differences\nin feature vectors cosine distance':[morphClosestRandomPerc[0][0],
                                                                                                        morphoClosestMorphPerc[0][0],
                                                                                                        morphClosestNucleiPerc[0][0],
                                                                                                        morphClosestVasculaturePerc[0][0],
                                                                                                        morphClosestEosinPerc[0][0]],
                                    'Top % of absolute differences\nin eosin intensity':[eosinClosestRandomPerc,
                                                                                        eosinClosestMorphPerc,
                                                                                        eosinClosestNucleiPerc,
                                                                                        eosinClosestVasculaturePerc,
                                                                                        eosinClosestEosinPerc]})

            allPairsDf = allPairsDf.append(pairsDf)
    if runCal:
        allPairsDf.to_csv(os.path.join(figuresDir, 'nucleiFeatures.csv'))

# %%

def __main__(saveFig=False, runCal=False):

    inputSamplesList = pd.read_csv(files['DATASETS']['cd31'])['Name'].tolist()
    featuresOfInterest = [nfeat.Location(), nfeat.Size()]

    if runCal:
        handcrafted_features_analysis(inputSamplesList,featuresOfInterest,saveFig,runCal=True)

    savedPairsDf = pd.read_csv(os.path.join(figuresDir, 'nucleiFeatures.csv'),index_col=[0])

    #*#*# VISUALIZATION #*#*#
    slideToVisualize = slidesToVisualize[slidesToVisualize['Reason']=='Handcrafted']['Name'].tolist()
    for name in slideToVisualize:

        chosenFeaturesDiff = handcrafted_features_analysis([name],featuresOfInterest,saveFig=saveFig,runCal=False)

        # sort by nuclei size:
        featPairsDf = savedPairsDf[['Name', 'Top % of absolute differences\nin nuclei area','Points chosen']]
        featPairsDf = featPairsDf[featPairsDf['Name'] == int(name)]
        featPairsDf = featPairsDf.drop('Name',axis=1)
        featPairsDf['Area incides'] =  featPairsDf['Top % of absolute differences\nin nuclei area']/100*len(chosenFeaturesDiff['nucleiArea'])
        featPairsDf['Areas'] =  chosenFeaturesDiff['nucleiArea'][np.array(featPairsDf['Area incides'].values,dtype=int)]

        ranks = np.arange(1, len(chosenFeaturesDiff['nucleiArea']) + 1)
        featPairsDf['Ranks'] = np.array([ranks[np.searchsorted(chosenFeaturesDiff['nucleiArea'], v, side="left")] for v in featPairsDf['Areas']])
        featPairsDf['Scaled ranks'] = 100 * (featPairsDf['Ranks'].values - np.min(featPairsDf['Ranks'].values)) / (np.max(featPairsDf['Ranks'].values) - np.min(featPairsDf['Ranks'].values))

        fig, axs = plt.subplots(figsize=(6,5))

        # scatter plot for DataFrame values using ranks:
        labels = ['Randomly','Based on\neosin intensity','Based on\nvasculature density','Based on\nnuclei size','Based on\nmorphological similarity']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']
        for cnt,label in enumerate(labels):
            subset = featPairsDf[featPairsDf['Points chosen'] == label]
            axs.scatter(subset['Scaled ranks'], [label] * len(subset),linewidth=0.25, marker='|', s=100,color=colors[cnt])
            axs.scatter(np.mean(subset['Scaled ranks']), label,linewidth=0.25, s=50, label=label, color=colors[cnt])
            mean_value = subset['Scaled ranks'].mean()
            axs.text(mean_value, cnt + 0.35, f'$\mu$={mean_value:.0f}', va='center', ha='center', color=colors[cnt], fontsize=13)

        axs.set_xlabel('Scaled rank in nucleus size difference\nbetween the points')
        handles, labels = axs.get_legend_handles_labels()
        axs.set_yticks(labels)
        axs.tick_params(axis='y')
        axs.set_yticklabels(['Random', 'Eosin intensity', 'Vasculature density', 'Nucleus size', 'Morphological similarity'])
        axs.set_ylim((-1,5))
        axs.set_ylabel('Selection feature', rotation=270, labelpad=30)
        axs.yaxis.set_label_position("right")
        fig.patch.set_alpha(0.0)
        if saveFig:
            featPairsDf.to_csv(os.path.join(figuresDir, 'ranks_handcrafted_'+name+'.csv'))
            fig.savefig(os.path.join(figuresDir, 'ranks_handcrafted_'+name+'.png'),bbox_inches='tight')
            fig.savefig(os.path.join(figuresDir, 'ranks_handcrafted_'+name+'.svg'),format='svg',dpi=600,bbox_inches='tight')  
        plt.show()

    pairsDfNoNames = savedPairsDf.drop('Name',axis=1)
    pairsDfNoNames = pairsDfNoNames.drop('Top % of absolute differences\nin feature vectors cosine distance',axis=1)

    pairsMean = pairsDfNoNames.groupby('Points chosen').mean()
    featuresOrder = ['Top % of absolute differences\nin nuclei area',
                    'Top % of absolute differences\nin vascular density',
                    'Top % of absolute differences\nin eosin intensity']
    pairsMean = pairsMean[featuresOrder]
    choiceOrder = ['Based on\nmorphological similarity',
                'Based on\nnuclei size',
                'Based on\nvasculature density',
                'Based on\neosin intensity',
                'Randomly']
        
    pairsMean = pairsMean.reindex(choiceOrder)

    pairsMean['Top % of absolute differences\nin eosin intensity'].loc['Based on\neosin intensity']=np.nan
    pairsMean['Top % of absolute differences\nin vascular density'].loc['Based on\nvasculature density']=np.nan
    pairsMean['Top % of absolute differences\nin nuclei area'].loc['Based on\nnuclei size']=np.nan

    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(pairsMean, annot=True)
    plt.xticks(rotation=0)  #
    plt.ylabel('Selection feature')
    plt.xlabel('\nComparison feature')
    plt.title('Scaled rank in feature similarity between\nchosen points averaged across all slides\n')
    ax.set_yticklabels(['Morphological\nsimilarity', 'Nucleus\nsize', 'Vasculature\ndensity','Eosin\nintensity','Random'])
    ax.set_xticklabels(['Nucleus\nsize','Vasculature\ndensity','Eosin\nintensity'])
    fig.patch.set_alpha(0.0)
    if saveFig:
        pairsMean.to_csv(os.path.join(figuresDir, 'ranks_handcrafted_avg.png'))
        fig.savefig(os.path.join(figuresDir, 'ranks_handcrafted_avg.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figuresDir, 'ranks_handcrafted_avg.svg'),format='svg',dpi=600,bbox_inches='tight')  
    plt.show()

# %%

if __name__ == "__main__":
    
    __main__()
