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

import scipy
import os
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import openslide as oSlide
import numpy as np

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import FuncFormatter
from numpy.random import permutation

os.chdir(os.path.dirname(os.path.dirname(__file__)))

import Utils.Image_Utils as iu
from Utils.Load_PatientInfo import load_patients_info
from Utils.Load_Vectors import load_vectors
from Utils.Load_Patches import load_patches_pilot
from Utils.Visualization import apply_plot_settings

with open('Utils/Global_Params.yaml') as file:
    files = yaml.full_load(file) 

# %% Visualization settings, helpe functions and color palettes.

slidesToVisualize = pd.read_csv(files['VISUALIZATION']['slides'])
figuresDir = files['FIGURES']
apply_plot_settings()

patientList = ['KC01138', 'KC01888','KC02820']
patientDir = {'KC01138':'Patient A', 'KC01888':'Patient B', 'KC02820':'Patient C'}

gradeColorDict = {1: 'darkgreen', 2: 'lightgreen', 3: 'orange', 4: 'red'}

pal = sns.color_palette("husl", 11)
archColorDict = {'Alveolar': 'blue', 'Large nest': pal[4], 'Small nest': pal[2],
                 'Macrocystic': pal[0], 'Papillary': pal[7], 'Solid': pal[8],
                 'Trabecular': pal[6], 'Solid-trabecular': pal[9], 'Bleeding follicles': pal[10]}

archCombinedDict = {'Small nest':1, 'Macrocystic':1, 'Large nest':1, 'Bleeding follicles':1,
                    'Alveolar':2, 'Trabecular':2, 'Papillary':2,
                    'Solid':3, 'Rhabdoid':3}

geneticOrderOnTree = {'KC01138':['T1a-A9','T1f-A3','T1e-A3','T1c-A4','T1b-A4','T1d-A3','M3b-A4','M3a-A12'],
                      'KC01888':['T1b-A7','M2a-B6','M1a-B5','Th1a-A3','T1c-A7','T1e-A5','T1d-A5'],
                      'KC02820':['T1a-A6','T1e-A11','T1c-A7','T1b-A7','Th1a-A5','T1f-A11','T1d-A9','M1b-A18','M2a-A13','M2b-A13','T1g-A15']}


# %% Helper functions.

def correct_lut(palette,patientName):
    """Adjust colors by switching their order for chosen samples.

    Args:
        palette (dictionary): colors (values) assigned to each sample (keys).
        patientName (str): patient name.

    Returns:
        palette (dictionary): colors (values) assigned to each sample (keys), correceted.
    """

    paletteOrg = palette.copy()
    if patientName == 'KC02820':
        palette['T1g-A15']=paletteOrg['T1a-A6']
        palette['T1a-A6']=paletteOrg['T1d-A9']
        palette['T1d-A9']=paletteOrg['T1f-A11']
        palette['T1f-A11']=paletteOrg['T1g-A15']
    elif patientName == 'KC01888':
        palette['T1c-A7']=paletteOrg['Th1a-A3']
        palette['Th1a-A3']=paletteOrg['T1c-A7']
    elif patientName == 'KC01138':
        palette['T1c-A4']=paletteOrg['T1d-A3']
        palette['T1d-A3']=paletteOrg['T1c-A4']

    return palette

def svs_annotation_files_search(svsId):
    """Fetches WSIs and their annotations paths.

    Args:
        svsId (str): name of the slide.

    Returns:
        svsFull (str): full path to the slide.
        annoFull (str): full path to the slide's annotations.
    """

    baseSvsDir=files['SLIDES']['pilot']
    baseAnnoDir=files['MASKS']['annotations_samples']
    batchSvsDirs=['Batch1/HnE/','Batch2/HnE','Batch3']
    batchAnnoDirs=['PilotBatch1_2.2_v3', 'QP22-Pilot_Batch2', 'QP22_Pilot_Batch3']

    svsFull = None
    annoFull = None

    for cnt in range(len(batchSvsDirs)):
        svsPalh=os.path.join(baseSvsDir, batchSvsDirs[cnt],svsId+'.svs')
        if os.path.exists(svsPalh):
            svsFull = svsPalh
            break

    for cnt2 in range(len(batchAnnoDirs)):
        annoPath=os.path.join(baseAnnoDir, batchAnnoDirs[cnt2], 'annotations', svsId+'.txt')
        if os.path.exists(annoPath):
            annoFull = annoPath
            break

    return svsFull, annoFull

def mantel_stratified(distDict, n_perm=9999):
    """Mantel correlation calculations to account for dependent distances.

    Args:
        distDict (dictionary):] distance matrices of cosine distances (value: [D1, D2]) for each patient (key).

    Returns:
        r_value (float): r-value.
        p_values (float): p-value.
    """

    v1_list, v2_list = [], []
    for patient, (D1, D2) in distDict.items():
        v1_list.append(D1[np.triu_indices_from(D1,k=1)])
        v2_list.append(D2[np.triu_indices_from(D2,k=1)])

    v1_obs = np.concatenate(v1_list)
    v2_obs = np.concatenate(v2_list)
    r_obs  = pearsonr(v1_obs, v2_obs)[0]

    greater = 0
    for p in range(n_perm):
        perm_v2 = []
        for D2 in (item[1] for item in distDict.values()):
            perm = permutation(D2.shape[0])
            perm_v2.append(D2[perm][:, perm][np.triu_indices_from(D2[perm][:, perm],k=1)])
        v2_perm = np.concatenate(perm_v2)
        if abs(pearsonr(v1_obs, v2_perm)[0]) >= abs(r_obs):
            greater += 1

    p_val = (greater + 1) / (n_perm + 1)
    return r_obs, p_val

def get_accuracies(vectors, labels, patient, samplePal, saveFig=False):
    """Calculate separation score between the different samples and saves the results in a heatmap form.

    Args:
        vectors (array): feature profiles. 
        labels (array): sample names corresponding to each feature profile (vetor).
        patient (str): patient name.
        samplePal (dictionary): colors (values) assigned to each sample (keys), correceted.
        saveFig (bool, optional): allows for saving figures as .svg and .png. Defaults to False.
    """

    uniqueClasses = geneticOrderOnTree[patient]
    nClasses = len(uniqueClasses)
    
    accuracyMatrix = np.zeros((nClasses, nClasses))
    for i, class_i in enumerate(uniqueClasses):
        for j, class_j in enumerate(uniqueClasses):
            if i == j:
                accuracyMatrix[i, j] = np.nan
                continue
            
            mask = (labels == class_i) | (labels == class_j)
            filteredVectors = vectors[mask]
            filteredLabels = labels[mask]
            
            x_train, x_test, y_train, y_test = train_test_split(filteredVectors, filteredLabels, test_size=0.5, random_state=111, stratify=filteredLabels, shuffle=True)
            model = SVC(kernel='linear', class_weight='balanced').fit(x_train, y_train)
            predictions = model.predict(x_test)
            
            accuracy = accuracy_score(y_test, predictions)
            accuracyMatrix[i, j] = accuracy

    annoMatrix = np.full_like(accuracyMatrix, '', dtype=object)
    for i in range(nClasses):
        for j in range(nClasses):
            if i <= j: 
                annoMatrix[i, j] = f"{accuracyMatrix[i, j]:.2f}"
    
    fig = plt.figure(figsize=(6,5))
    g = sns.heatmap(accuracyMatrix, annot=annoMatrix, fmt="", cmap="coolwarm", xticklabels=uniqueClasses, yticklabels=uniqueClasses, cbar_kws={'label': 'Separation'},annot_kws={"size": 10})

    for xtick in g.get_xticklabels():
        xtick.set_color(samplePal[xtick.get_text()])
        xtick.set_fontweight("bold")

    for ytick in g.get_yticklabels():
        ytick.set_color(samplePal[ytick.get_text()])
        ytick.set_fontweight("bold")

    plt.title('Separation heatmap ('+patientDir[patient]+')\n')
    fig.patch.set_alpha(0.0)
    if saveFig:
        fig.savefig(os.path.join(figuresDir, patient+'_sep_heatmap.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figuresDir, patient+'_sep_heatmap.svg'),format='svg',dpi=600, bbox_inches='tight')
    plt.show()

# %% # PER-PATCH REPRESENTATIONS #

def per_patch_analysis(patientList, patchSaveDir, dataDir, saveFig=False):
    """Per-patch analysis of morphology as derived by MorphoITH.

    Args:
        patientList (list): list of patient names.
        patchSaveDir (str): full path to where the patches are saved.
        dataDir (dictionary): feature profiles (values) for each sample (keys).
        saveFig (bool, optional): allows for saving figures as .svg and .png. Defaults to False.
    """

    for patientName in patientList:

        allSamplesInfo = load_patients_info()
        allSamplesInfo = allSamplesInfo.set_index('Patient.ID')
        patientInfo = allSamplesInfo.loc[patientName]
        patientInfo = patientInfo[patientInfo.Location != 'Normal']

        patchesDir = load_patches_pilot([patientName], patchSaveDir)
        patches = np.concatenate([patchesDir[sample] for sample in patientInfo['Sample.ID']])

        profilesMat = np.concatenate([dataDir[sample] for sample in patientInfo['Sample.ID']])
        data = TSNE(init='random',n_components=2, metric='cosine', perplexity=600, random_state=20, learning_rate=200,square_distances=True).fit_transform(profilesMat)

        shortIdsAll = np.concatenate([np.array([sample.split('-', 1)[1]] * len(dataDir[sample])) for sample in patientInfo['Sample.ID']])
        shortIds = np.asarray([sample.split('-', 1)[1] for sample in patientInfo['Sample.ID'].values])

        cmap = plt.cm.get_cmap('tab20b', shortIds.shape[0])
        samplePal = dict(zip(sorted(shortIds), [cmap(i) for i in range(shortIds.shape[0])]))
        samplePal = correct_lut(samplePal,patientName)
        
        cmap = plt.cm.get_cmap('Set2', patientInfo['Image.Flip'].unique().shape[0])
        svsPal = dict(zip(sorted(patientInfo['Image.Flip'].unique()), [cmap(i) for i in range(patientInfo['Image.Flip'].unique().shape[0])]))

        profilesPcaSubset = PCA(n_components=10).fit_transform(profilesMat)
        get_accuracies(profilesPcaSubset, shortIdsAll, patientName, samplePal)
        
        allGrades = []
        allArchs = []
        allSlides = []

        markersDict = {'M':(5,1),'T':'o'}
        fig = plt.figure(figsize=(5,5))
        for cnt, sample in enumerate(shortIds):
            isInSample = shortIdsAll == sample
            plt.scatter(data[isInSample, 0], data[isInSample, 1], label=sample,color=samplePal[sample],marker=markersDict[sample[0]])
            
        labels = geneticOrderOnTree[patientName]
        custom_handles = [plt.scatter([], [], color=samplePal[sample], s=500, marker=markersDict[sample[0]]) for sample in labels]
        plt.legend(custom_handles, labels, bbox_to_anchor=(1.05, 1.03), loc='upper left',framealpha=1.0, facecolor='white')
        plt.title('Samples ('+patientDir[patientName]+')\n')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.xticks([],[])
        plt.yticks([],[])
        fig.patch.set_alpha(0.0)
        if saveFig:
            fig.savefig(os.path.join(figuresDir, 'scatterplot_sample_'+patientName+'.png'), bbox_inches='tight')
            fig.savefig(os.path.join(figuresDir, 'scatterplot_sample_'+patientName+'.svg'),format='svg',dpi=600, bbox_inches='tight')
        plt.show()

        # grade:
        fig = plt.figure(figsize=(5,5))

        for cnt, sample in enumerate(shortIds):
            isInSample = shortIdsAll == sample
            grade = patientInfo[patientInfo['Sample.ID']==patientName+'-'+sample]['Grade- Punch'].item()
            allGrades.append([grade]*isInSample.sum())
            plt.scatter(data[isInSample, 0], data[isInSample, 1], 
                        label=grade,
                        marker=markersDict[sample[0]],
                        color=gradeColorDict[int(grade)])
        labels = np.sort( np.unique(patientInfo['Grade- Punch']))
        labels = labels.astype(np.uint8)
        custom_handles = [plt.scatter([], [], color=gradeColorDict[grade], s=300) for grade in labels]
        plt.legend(custom_handles, labels, bbox_to_anchor=(1.05, 1.03), loc='upper left',framealpha=1.0, facecolor='white')

        plt.title('Grades ('+patientDir[patientName]+')\n')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.xticks([],[])
        plt.yticks([],[])
        fig.patch.set_alpha(0.0)
        if saveFig:
            fig.savefig(os.path.join(figuresDir, 'scatterplot_grade_'+patientName+'.png'), bbox_inches='tight')
            fig.savefig(os.path.join(figuresDir, 'scatterplot_grade_'+patientName+'.svg'),format='svg',dpi=600, bbox_inches='tight')
        plt.show()

        # architecture:
        fig = plt.figure(figsize=(5,5))

        for cnt, sample in enumerate(shortIds):
            isInSample = shortIdsAll == sample
            arch = patientInfo[patientInfo['Sample.ID']==patientName+'-'+sample]['Pattern.Uniformity.2'].item()
            allArchs.append([arch]*isInSample.sum())
            plt.scatter(data[isInSample, 0], data[isInSample, 1], 
                        label=grade,
                        marker=markersDict[sample[0]],
                        color=archColorDict[str(arch)])
        labels = np.sort( np.unique(patientInfo['Pattern.Uniformity.2']))
        custom_handles = [plt.scatter([], [], color=archColorDict[str(grade)], s=300) for grade in labels]
        plt.legend(custom_handles, labels, bbox_to_anchor=(1.05, 1.03), loc='upper left',framealpha=1.0, facecolor='white')

        plt.title('Architectures ('+patientDir[patientName]+')\n')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.xticks([],[])
        plt.yticks([],[])
        fig.patch.set_alpha(0.0)
        if saveFig:
            fig.savefig(os.path.join(figuresDir, 'scatterplot_arch_'+patientName+'.png'), bbox_inches='tight')
            fig.savefig(os.path.join(figuresDir, 'scatterplot_arch_'+patientName+'.svg'),format='svg',dpi=600, bbox_inches='tight')
        plt.show()

        # slide:
        fig = plt.figure(figsize=(5,5))

        for cnt, sample in enumerate(shortIds):
            isInSample = shortIdsAll == sample
            svsName = patientInfo[patientInfo['Sample.ID']==patientName+'-'+sample]['Image.Flip'].item()
            allSlides.append([svsName]*isInSample.sum())
            plt.scatter(data[isInSample, 0], data[isInSample, 1], 
                        label=svsName,
                        marker=markersDict[shortIds[cnt][0]],
                        color=svsPal[svsName])
        originalLabels = np.sort(np.unique(patientInfo['Image.Flip']))
        labels = []
        for label in originalLabels:
            _,part2,part3,_ = label.split('-')
            labels.append(part2+'-'+part3)
        custom_handles = [plt.scatter([], [], color=svsPal[svsName], s=300) for svsName in originalLabels]
        plt.legend(custom_handles, labels, bbox_to_anchor=(1.05, 1.03), loc='upper left',framealpha=1.0, facecolor='white')

        plt.title('Slides ('+patientDir[patientName]+')\n')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.xticks([],[])
        plt.yticks([],[])
        fig.patch.set_alpha(0.0)
        if saveFig:
            pd.DataFrame({'t-SNE 1':data[:,0],'t-SNE 2':data[:,1],'Sample':shortIdsAll, 'Slide':np.concatenate(allSlides), 'Grade':np.concatenate(allGrades), 'Architecture':np.concatenate(allArchs)}).to_csv(os.path.join(figuresDir, 'scatterplot_'+patientName+'.csv'))
            fig.savefig(os.path.join(figuresDir, 'scatterplot_slide_'+patientName+'.png'), bbox_inches='tight')
            fig.savefig(os.path.join(figuresDir, 'scatterplot_slide_'+patientName+'.svg'),format='svg',dpi=600, bbox_inches='tight')
        plt.show()

        # overlay patches:
        fig, ax = plt.subplots(figsize=(9,9))
        for cnt, sample in enumerate(shortIds):
                
            isInSample = shortIdsAll == sample
            plt.scatter(data[isInSample, 0], data[isInSample, 1], label=sample,color=samplePal[sample])
            single_data=data[isInSample]
            single_patches=patches[isInSample]
            for cnt2, (x0, y0) in enumerate(zip(single_data[:,0], single_data[:,1])):
                ab = AnnotationBbox(OffsetImage(np.uint8(single_patches[cnt2]),zoom=0.1), (x0, y0), 
                bboxprops=dict(edgecolor=samplePal[sample], linewidth=4),pad=0.01,box_alignment=(0.5, 0.5)) 
                ax.add_artist(ab)
            ax.autoscale()
            
        labels = geneticOrderOnTree[patientName]
        custom_handles = [plt.scatter([], [], color=samplePal[sample], s=300, marker=markersDict[sample[0]]) for sample in labels]
        plt.legend(custom_handles, labels, bbox_to_anchor=(1.05, 1.02), loc='upper left',framealpha=1.0, facecolor='white')
        plt.title('Samples ('+patientDir[patientName]+')\n')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.xticks([],[])
        plt.yticks([],[])
        fig.patch.set_alpha(0.0)
        if saveFig:
            fig.savefig(os.path.join(figuresDir, 'scatterplot_patches_'+patientName+'.png'), bbox_inches='tight')
            fig.savefig(os.path.join(figuresDir, 'scatterplot_patches_'+patientName+'.svg'),format='svg',dpi=600, bbox_inches='tight')
        plt.show()


# %% # MEAN REPRESENTATIONS #

def per_representation_analysis(patientList, dataDir, normalization = False, normalizationScheme = '', saveFig = False, uni=False):
    """Performs correlation between genetic and morphologic ITH.

    Args:
        patientList (list): list of patient names.
        dataDir (dictionary): feature profiles (values) for each sample (keys).
        normalization (bool, optional): allows for analysis on normalized patches. Defaults to False.
        normalizationScheme (str, optional): name of normalization scheme (Macenko or Vahadane). Defaults to ''.
        saveFig (bool, optional): allows for saving figures as .svg and .png. Defaults to False.
    """      

    corrPd = pd.DataFrame()
    corrDict = {}

    for patientName in patientList:

        allSamplesInfo = load_patients_info()
        allSamplesInfo = allSamplesInfo.set_index('Patient.ID')
        patientInfo = allSamplesInfo.loc[patientName]
        patientInfo = patientInfo[patientInfo.Location != 'Normal']
        
        shortIds = np.asarray([sample.split('-', 1)[1] for sample in patientInfo['Sample.ID'].values])

        cmap = plt.cm.get_cmap('tab20b', shortIds.shape[0])
        samplePal = dict(zip(sorted(shortIds), [cmap(i) for i in range(shortIds.shape[0])]))
        samplePal = correct_lut(samplePal,patientName)

        centroidsMat = np.vstack([np.mean(dataDir[sample], axis=0) for sample in patientInfo['Sample.ID']])
        sparseMatCentroids = scipy.sparse.csr_matrix(centroidsMat)
        distanceMatCentroids = cosine_distances(sparseMatCentroids)
        idx1, idx2  = np.triu_indices_from(distanceMatCentroids, k=1)
        distanceMatCentroidsUpper = distanceMatCentroids[idx1,idx2]

        grades = patientInfo['Grade- Punch'].values
        archs = patientInfo['Pattern.Uniformity.2'].values
        archsCombined = np.asarray([archCombinedDict[i] for i in archs])
        archsPairs = []
        gradesPairs = []
        archsCombinedPairs = []
        for i, j in zip(*(idx1,idx2)):
            if grades[i] < grades[j]:
                gradesPairs.append(str(int(grades[i]))+'-'+str(int(grades[j])))
            else:
                gradesPairs.append(str(int(grades[j]))+'-'+str(int(grades[i])))
            if archs[i] < archs[j]:
                archsPairs.append(str(archs[i])+'-'+str(archs[j]))
            else:
                archsPairs.append(str(archs[j])+'-'+str(archs[i]))
            if archsCombined[i] < archsCombined[j]:
                archsCombinedPairs.append(str(archsCombined[i])+'-'+str(archsCombined[j]))
            else:
                archsCombinedPairs.append(str(archsCombined[j])+'-'+str(archsCombined[i]))

        geneticDistancesPd = pd.read_csv(os.path.join(files['DATA']['phylogenetic'],'distances_'+patientName+'.csv'),index_col=[0])
        geneticDistancesPd = geneticDistancesPd.loc[shortIds, shortIds]
        geneticDistances = geneticDistancesPd.values
        idx1, idx2 = np.triu_indices_from(geneticDistances, k=1)
        geneticDistancesUpper = geneticDistances[idx1,idx2]

        corrDict[patientName] = [geneticDistances,distanceMatCentroids]

        corrNames = [(geneticDistancesPd.index[i], geneticDistancesPd.columns[j]) for i, j in zip(*(idx1,idx2))]

        corrSinglePd = pd.DataFrame({'MorphoITH distance':distanceMatCentroidsUpper,
                                     'Genetic distance':geneticDistancesUpper,
                                     'Label':[patientDir[patientName]]*(distanceMatCentroidsUpper.shape[0]),
                                     'Pairs':corrNames,
                                     'Grade transition':gradesPairs,
                                     'Architecture class transition':archsCombinedPairs,
                                     'Patient':[patientName]*(distanceMatCentroidsUpper.shape[0])})

        corrPd = corrPd.append(corrSinglePd)

        if (not normalization) & (not uni):

            distanceMatCentroidsPd = pd.DataFrame(distanceMatCentroids,index=shortIds,columns=shortIds)
            distanceMatCentroidsPd = distanceMatCentroidsPd.loc[geneticOrderOnTree[patientName], geneticOrderOnTree[patientName]]

            g = sns.clustermap(distanceMatCentroidsPd, yticklabels=distanceMatCentroidsPd.index, xticklabels=False,
                               cbar_kws={'label': 'MorphoITH distance'}, cmap='Greys_r',row_cluster=False,col_cluster=False,
                               figsize=(5,4),linewidths=0.5, cbar_pos=(.95, .2, 0.03, 0.4))
            g.fig.patch.set_alpha(0.0)
            g.cax.tick_params(labelsize=10)
            g.cax.yaxis.label.set_size(12)

            for ytick in g.ax_heatmap.get_yticklabels():
                ytick.set_color(samplePal[ytick.get_text()])
                ytick.set_fontweight("bold")
                ytick.set_fontsize(12)

            if saveFig:
                pd.DataFrame(distanceMatCentroids,index=shortIds,columns=shortIds).to_csv(os.path.join(figuresDir, 'heatmap_distance_'+patientName+'.csv'))
                g.savefig(os.path.join(figuresDir, 'heatmap_distance_'+patientName+'.png'), bbox_inches='tight')
                g.savefig(os.path.join(figuresDir, 'heatmap_distance_'+patientName+'.svg'),format='svg',dpi=600, bbox_inches='tight')
            plt.show()

    corrPd = corrPd.reset_index(drop=True)

    if not normalization:
        # genetics vs morphology:
        r_value, p_value = mantel_stratified(corrDict)

        fig = plt.figure(figsize=(5,5))
        sns.scatterplot(data=corrPd, x='MorphoITH distance', y='Genetic distance', hue='Label', s=60, palette='Set2')
        if p_value == 0.0001:
            plt.text(0.05, .65, f'r = {r_value:.3f},\np < {p_value:.1e}', transform=plt.gca().transAxes, fontsize=12, color='grey')
        else:
            plt.text(0.05, .65, f'r = {r_value:.3f},\np = {p_value:.1e}', transform=plt.gca().transAxes, fontsize=12, color='grey')
        plt.legend(loc='upper left',fontsize=13)
        fig.patch.set_alpha(0.0)
        if saveFig:
            corrPd[['Label','MorphoITH distance','Genetic distance']].to_csv(os.path.join(figuresDir, 'correlation_morpho.csv'))
            fig.savefig(os.path.join(figuresDir, 'correlation_morpho_3val.png'), bbox_inches='tight')
            fig.savefig(os.path.join(figuresDir, 'correlation_morpho_3val.svg'),format='svg',dpi=600, bbox_inches='tight')
        plt.show()
        fig = plt.figure(figsize=(5,5))
        sns.scatterplot(data=corrPd, x='MorphoITH distance', y='Genetic distance', hue='Label', s=60, palette='Set2')
        if p_value == 0.0001:
            plt.text(0.05, .65, f'r = {r_value:.2f},\np < {p_value:.1e}', transform=plt.gca().transAxes, fontsize=12, color='grey')
        else:
            plt.text(0.05, .65, f'r = {r_value:.2f},\np = {p_value:.1e}', transform=plt.gca().transAxes, fontsize=12, color='grey')
        plt.legend(loc='upper left',fontsize=13)
        fig.patch.set_alpha(0.0)
        if saveFig:
            corrPd[['Label','MorphoITH distance','Genetic distance']].to_csv(os.path.join(figuresDir, 'correlation_morpho.csv'))
            fig.savefig(os.path.join(figuresDir, 'correlation_morpho.png'), bbox_inches='tight')
            fig.savefig(os.path.join(figuresDir, 'correlation_morpho.svg'),format='svg',dpi=600, bbox_inches='tight')
        plt.show()
        if not uni:
            # genetics vs grade/architecture:
            fig, axes  = plt.subplots(nrows=2,ncols=1,figsize=(7,5))
            gradeOrder=['1-1','2-2','3-3','1-2','2-3','1-3',]
            archsOrder=['1-1','2-2','3-3','1-2','2-3','1-3']
            patientOrder=['Patient A', 'Patient B', 'Patient C']

            sns.swarmplot(corrPd,x='Grade transition',y='Genetic distance',order=gradeOrder,hue='Label',hue_order=patientOrder,ax=axes[0], palette='Set2',size=4)
            axes[0].set_title("")
            axes[0].set_ylabel("")
            axes[0].set_xlabel("Nuclear grade transition")
            axes[0].legend(loc="upper left", bbox_to_anchor=(1, 1.1),framealpha=1.0, facecolor='white')

            sns.swarmplot(corrPd,x='Architecture class transition',y='Genetic distance',order=archsOrder,hue='Label',hue_order=patientOrder,ax=axes[1],legend=None, palette='Set2',size=4)
            axes[1].set_title("")
            axes[1].set_xlabel("Architecture class transitions")
            axes[1].set_ylabel("")
            fig.text(0.04, 0.5, 'Genetic distance\n', va='center', ha='center', rotation='vertical', fontsize=18)
            plt.tight_layout()
            fig.patch.set_alpha(0.0)
            if saveFig:
                corrPd[['Label','Grade transition','Architecture class transition','Genetic distance']].to_csv(os.path.join(figuresDir, 'correlation_archs_grade.csv'))
                fig.savefig(os.path.join(figuresDir, 'correlation_archs_grade.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, 'correlation_archs_grade.svg'),format='svg',dpi=600, bbox_inches='tight')
            plt.show()

    if not uni:
        # correlation per patient:
        labels = corrPd['Label'].unique()
        pal = sns.color_palette('Set2', len(labels))
        colors = dict(zip(labels, pal))

        fig, axes = plt.subplots(1, len(labels), figsize=(13, 5), sharey=True) 
        for ax, label in zip(axes, labels):
            subset = corrPd[corrPd['Label'] == label]
            patient = subset['Patient'].unique()[0]
            r_value, p_value = mantel_stratified({ patient: corrDict[patient] }) 

            sns.scatterplot(data=subset, x='MorphoITH distance', y='Genetic distance', s=60, color=colors[label], ax=ax, legend=False)
            ax.text(0.05, 0.95, f'r = {r_value:.2f}\np = {p_value:.1e}', transform=ax.transAxes, fontsize=14, color='grey',verticalalignment='top')
            ax.set_title(label)
            ax.set_xlabel('MorphoITH distance')
            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel('Genetic distance')
            else:
                ax.set_ylabel('') 
            formatter = FuncFormatter(lambda x, _: f'{x:.2f}')
            ax.xaxis.set_major_formatter(formatter)
        if normalization:
            print(normalizationScheme)
        plt.tight_layout()
        fig.patch.set_alpha(0.0)
        if saveFig:
            corrPd[['Label','MorphoITH distance','Genetic distance']].to_csv(os.path.join(figuresDir, 'correlation_morpho_perpatient'+normalizationScheme+'.csv'))
            fig.savefig(os.path.join(figuresDir, 'correlation_morpho_perpatient'+normalizationScheme+'.png'), bbox_inches='tight')
            fig.savefig(os.path.join(figuresDir, 'correlation_morpho_perpatient'+normalizationScheme+'.svg'),format='svg',dpi=600, bbox_inches='tight')
        plt.show()

# %%

def __main__(saveFig=False, norm=False, uni=False):

    if norm:
        for normScheme in ['Macenko','Vahadane']:
            patchSaveDir = os.path.join(files['DATA']['pilot']['cores_20x_224px'],'Normalized'+normScheme)
            profilesSaveDir = os.path.join(files['FEATURES']['morphoith'],'Pilot_ForPaper','Normalized'+normScheme)
            dataDir = load_vectors(profilesSaveDir)
            per_representation_analysis(patientList, dataDir, normalization=norm, normalizationScheme=normScheme, saveFig=saveFig)

    else:
        patchSaveDir = files['DATA']['pilot']['cores_20x_224px']
        profilesSaveDir = os.path.join(files['FEATURES']['morphoith'],'Pilot_ForPaper')
        if uni:
            profilesSaveDir = os.path.join(files['FEATURES']['uni'],'Pilot_ForPaper')
        dataDir = load_vectors(profilesSaveDir)

        per_representation_analysis(patientList, dataDir, normalization=norm, saveFig=saveFig,uni=uni)

        if not uni:
            per_patch_analysis(patientList, patchSaveDir, dataDir, saveFig=saveFig)

            # plot sample areas:
            patient = 'KC02820'
            for sample in ['T1e-A11','M2b-A13',
                        'T1g-A15','T1a-A6',
                        'M2a-A13','Th1a-A5']:

                fullSample = patient+'-'+sample
                allSamplesInfo = load_patients_info()
                imageName = allSamplesInfo[allSamplesInfo['Sample.ID']==fullSample]['Image.Flip'].item()
                punchId = allSamplesInfo[allSamplesInfo['Sample.ID']==fullSample]['Punch.ID'].item()
                svsFlip, annoFlip = svs_annotation_files_search(imageName.split('.')[0])

                slideFlip = oSlide.open_slide(str(svsFlip))

                __,regionNames,regionInfo,_ = iu.GetQPathTextAnno(str(annoFlip))
                regionNames = [p.strip('+') for p in regionNames]

                idx = next((i for i, x in enumerate(regionNames) if punchId in x and '_X2' in x), next((i for i, x in enumerate(regionNames) if punchId in x), None))

                x1F, y1F, x2F, y2F = regionInfo[idx]['BoundingBox']
                img = np.array(slideFlip.read_region((int(x1F), int(y1F)), 0, (int(x2F - x1F), int(y2F - y1F))))
                
                if sample == 'T1g-A15':
                    cutImg = img[4500:8500,1000:5000,:]
                elif sample == 'Th1a-A5':
                    cutImg = img[500:4500,500:4500,:]
                elif sample == 'T1e-A11':
                    cutImg = img[3000:7000,4500:8500,:]
                else:
                    cutImg = img[2500:6500,2500:6500,:]

                fig = plt.figure()
                plt.imshow(cutImg)
                plt.xticks([],[])
                plt.yticks([],[])
                plt.title(patientDir[patient]+' ('+sample+')')
                fig.patch.set_alpha(0.0) 
                if saveFig:
                    fig.savefig(os.path.join(figuresDir, 'punch_'+fullSample+'.png'), bbox_inches='tight')
                    fig.savefig(os.path.join(figuresDir, 'punch_'+fullSample+'.svg'),format='svg',dpi=600, bbox_inches='tight')
                plt.show()

            # check annotations sizes:
            annoSizes = []
            for patientName in patientList:

                allSamplesInfo = load_patients_info()
                allSamplesInfo = allSamplesInfo.set_index('Patient.ID')
                allSamplesInfo['Slide ID Flip'] = allSamplesInfo['Image.Flip'].apply(lambda x: x.split('.')[0])
                patientInfo = allSamplesInfo.loc[patientName]
                patientInfo = patientInfo[patientInfo.Location != 'Normal']

                imageNames  =  patientInfo['Image.Flip'].unique()

                for imageName in imageNames:
                    svsFlip, annoFlip = svs_annotation_files_search(imageName.split('.')[0])
                    slideFlip = oSlide.open_slide(str(svsFlip))
                    mpp = np.mean([float(slideFlip.properties[p]) for p in slideFlip.properties if 'mpp' in p.lower()])
                    _,regionNames,regionInfo,_ = iu.GetQPathTextAnno(str(annoFlip))
                    regionNames = [p.strip('+') for p in regionNames]

                    punchIds = patientInfo[patientInfo['Image.Flip']==imageName]['Punch.ID']

                    for punchId in punchIds:
                        idx = next((i for i, x in enumerate(regionNames) if punchId in x and '_X2' in x), next((i for i, x in enumerate(regionNames) if punchId in x), None))
                        areaPx = regionInfo[idx]['Area']
                        annoSizes.append((areaPx * (mpp ** 2)) / (1000 ** 2))

            print('Mean sample size:',np.mean(annoSizes))     

# %%

if __name__ == "__main__":
    
    __main__()
    
