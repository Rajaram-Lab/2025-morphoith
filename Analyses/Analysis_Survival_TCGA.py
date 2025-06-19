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
import yaml
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter
from lifelines.utils import ConvergenceWarning
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from statannot import add_stat_annotation

os.chdir(os.path.dirname(os.path.dirname(__file__)))
with open('Utils/Global_Params.yaml') as file:
    files = yaml.full_load(file) 

from Utils.Visualization import apply_plot_settings
apply_plot_settings()

figuresDir = files['FIGURES']

# %%

def het_vs_grade_mstage(mergeHetAndSur, saveFig):
    """Checks for correlation between the heterogeneity score and nuclear grade.

    Args:
        mergeHetAndSur (DataFrame): metadata with survival information for each patient.
        saveFig (bool): allows for saving figures as .svg and .png.
    """

    grades = mergeHetAndSur['Grade'].values

    gradeColorDict={'High':'orange', 'Low':'darkgreen'}
    gradesDict = {1:'Low',2:'Low',3:'High',4:'High'}
    stageColorDict = {'M0':'darkgreen','M1':'orange'}

    simpleGrades = [gradesDict[i] for i in grades]
    mergeHetAndSur['Grade simple'] = simpleGrades

    print('Heterogeneity vs grade')
    fig = plt.figure(figsize=(5,5))
    ax = sns.boxplot(data=mergeHetAndSur,x='Grade simple',y='Score',order=['Low','High'],palette=gradeColorDict)
    add_stat_annotation(ax=ax,data=mergeHetAndSur, x='Grade simple', y='Score',
                        box_pairs=[('Low','High')],
                        test='Mann-Whitney', text_format='simple', loc='inside', 
                        line_offset_to_box=0.1, verbose=0)
    plt.ylabel('Heterogeneity score')
    plt.xlabel('Nuclear grade')
    fig.patch.set_alpha(0.0)
    if saveFig:
        fig.savefig(os.path.join(figuresDir, 'het_grade.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figuresDir, 'het_grade.svg'),format='svg',dpi=600,bbox_inches='tight')
    plt.show()

    print('Heterogeneity vs mStage')
    fig = plt.figure(figsize=(5,5))
    ax = sns.boxplot(data=mergeHetAndSur,x='mStage',y='Score',palette=stageColorDict)
    add_stat_annotation(ax=ax,data=mergeHetAndSur, x='mStage', y='Score',
                        box_pairs=[('M0','M1')],
                        test='Mann-Whitney', text_format='simple', loc='inside', 
                        line_offset_to_box=0.1, verbose=0)
    plt.ylabel('Heterogeneity score')
    plt.xlabel('mStage')
    fig.patch.set_alpha(0.0)
    if saveFig:
        fig.savefig(os.path.join(figuresDir, 'het_mstage.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figuresDir, 'het_mstage.svg'),format='svg',dpi=600,bbox_inches='tight')
    plt.show()

    print('Heterogeneity vs mStage (high grade only)')
    fig = plt.figure(figsize=(5,5))
    ax = sns.boxplot(data=mergeHetAndSur[mergeHetAndSur['Grade']>=3],x='mStage',y='Score',palette=stageColorDict)
    add_stat_annotation(ax=ax,data=mergeHetAndSur[mergeHetAndSur['Grade']>=3], x='mStage', y='Score',
                        box_pairs=[('M0','M1')],
                        test='Mann-Whitney', text_format='simple', loc='inside', 
                        line_offset_to_box=0.1, verbose=0)
    plt.ylabel('Heterogeneity score')
    plt.xlabel('mStage')
    fig.patch.set_alpha(0.0)
    if saveFig:
        fig.savefig(os.path.join(figuresDir, 'het_mstage_highGrade.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figuresDir, 'het_mstage_highGrade.svg'),format='svg',dpi=600,bbox_inches='tight')
    plt.show()


def c_index(mergeHetAndSur):
    """Generates c-index values for nuclear grade alone, or nuclear grade and heterogeneity score together.

    Args:
        mergeHetAndSur (DataFrame): metadata with survival information for each patient.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)

        cph = CoxPHFitter()
        cph.fit(mergeHetAndSur[['Survival times', 'Censoring values', 'Grade']], duration_col='Survival times', event_col='Censoring values')
        c_index_grade = cph.concordance_index_

        cph = CoxPHFitter()
        cph.fit(mergeHetAndSur[['Survival times', 'Censoring values', 'Heterogeneity', 'Grade']], duration_col='Survival times', event_col='Censoring values')
        c_index_full = cph.concordance_index_

        print("C-index with grade only:", c_index_grade)
        print("C-index with grade + heterogeneity:", c_index_full)

        print(cph.summary)


def survival_plot(mergeHetAndSur, saveFig, n_clusters):
    """Plots fitting the Kaplan-Meier estimate for the survival function.

    Args:
        mergeHetAndSur (DataFrame): metadata with survival information for each patient.
        saveFig (bool): allows for saving figures as .svg and .png.
        n_clusters (int): number of morphological clusters used.
    """

    mergeHetAndSurStrat = mergeHetAndSur.copy()
    median_het = mergeHetAndSurStrat['Score'].median()
    mergeHetAndSurStrat['Heterogeneity group'] = mergeHetAndSurStrat['Score'] >= median_het

    kmf = KaplanMeierFitter()

    fig = plt.figure(figsize=(5,5))

    kmf.fit(durations=mergeHetAndSurStrat.loc[~mergeHetAndSurStrat['Heterogeneity group'], 'Survival times'],
            event_observed=mergeHetAndSurStrat.loc[~mergeHetAndSurStrat['Heterogeneity group'], 'Censoring values'],
            label='Low heterogeneity')
    ax = kmf.plot_survival_function(ci_show=True)

    kmf.fit(durations=mergeHetAndSurStrat.loc[mergeHetAndSurStrat['Heterogeneity group'], 'Survival times'],
            event_observed=mergeHetAndSurStrat.loc[mergeHetAndSurStrat['Heterogeneity group'], 'Censoring values'],
            label='High heterogeneity')
    kmf.plot_survival_function(ax=ax, ci_show=True)

    results=logrank_test(durations_A=mergeHetAndSurStrat.loc[mergeHetAndSurStrat['Heterogeneity group'], 'Survival times'],
                        durations_B=mergeHetAndSurStrat.loc[~mergeHetAndSurStrat['Heterogeneity group'], 'Survival times'],
                        event_observed_A=mergeHetAndSurStrat.loc[mergeHetAndSurStrat['Heterogeneity group'], 'Censoring values'], 
                        event_observed_B=mergeHetAndSurStrat.loc[~mergeHetAndSurStrat['Heterogeneity group'], 'Censoring values'])

    print("log-rank p-value: {:.3}".format(results.p_value))

    ax.text(0.25, 0.95, f'long-rank p-value = {results.p_value:.3f}', transform=ax.transAxes, fontsize=14, color='black',verticalalignment='top')
    plt.xlabel('Months')
    plt.ylabel('Survival probability')
    plt.legend()
    plt.grid(True)
    fig.patch.set_alpha(0.0)
    plt.tight_layout()
    if saveFig:
        if n_clusters == 10:
            fig.savefig(os.path.join(figuresDir, 'survival_tcga.png'), bbox_inches='tight')
            fig.savefig(os.path.join(figuresDir, 'survival_tcga.svg'),format='svg',dpi=600,bbox_inches='tight')
        else:
            fig.savefig(os.path.join(figuresDir, 'survival_tcga_clusters'+str(n_clusters)+'.png'), bbox_inches='tight')
            fig.savefig(os.path.join(figuresDir, 'survival_tcga_clusters'+str(n_clusters)+'.svg'),format='svg',dpi=600,bbox_inches='tight')     
    plt.show()

    # for all combinations:
    print('Combination of heterogeneity and grade')
    grades = mergeHetAndSur['Grade'].values
    gradesDict = {1:'Low',2:'Low',3:'High',4:'High'}

    simpleGrades = [gradesDict[i] for i in grades]
    mergeHetAndSurStrat['Grade simple'] = simpleGrades

    gradesToPlot = sorted(mergeHetAndSurStrat['Grade simple'].unique())

    palette = sns.color_palette() 
    color_map = {
        False: palette[0],
        True:  palette[1],
    }
    style_map = {gradesToPlot[0]: 'dashed', gradesToPlot[1]: 'solid'}

    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(5,5))

    for grade in gradesToPlot:
        for het_flag in [False, True]:
            mask = ((mergeHetAndSurStrat['Grade simple'] == grade) & (mergeHetAndSurStrat['Heterogeneity group'] == het_flag))
            durations = mergeHetAndSurStrat.loc[mask, 'Survival times']
            events = mergeHetAndSurStrat.loc[mask, 'Censoring values']

            kmf.fit(durations=durations,
                    event_observed=events,
                    label=f'{grade} grade – {"High" if het_flag else "Low"} heterogeneity')
            kmf.plot_survival_function(ax=ax, ci_show=False, color=color_map[het_flag], linestyle=style_map[grade])

    ax.set_xlabel('Months')
    ax.set_ylabel('Survival probability')
    ax.legend(frameon=True,fontsize=12.5)
    plt.grid(True)
    plt.tight_layout()
    fig.patch.set_alpha(0.0)
    if saveFig:
        if n_clusters == 10:
            fig.savefig(os.path.join(figuresDir, 'survival_wgrade_tcga.png'), bbox_inches='tight')
            fig.savefig(os.path.join(figuresDir, 'survival_wgrade_tcga.svg'),format='svg',dpi=600,bbox_inches='tight')
    plt.show()

    # only for grades 1-2:
    print('Low grade only')
    mergeHetAndSurStrat = mergeHetAndSurStrat[mergeHetAndSurStrat['Grade'] <3]

    kmf = KaplanMeierFitter()

    fig = plt.figure(figsize=(5,5))

    kmf.fit(durations=mergeHetAndSurStrat.loc[~mergeHetAndSurStrat['Heterogeneity group'], 'Survival times'],
            event_observed=mergeHetAndSurStrat.loc[~mergeHetAndSurStrat['Heterogeneity group'], 'Censoring values'],
            label='Low heterogeneity')
    ax = kmf.plot_survival_function(ci_show=True)

    kmf.fit(durations=mergeHetAndSurStrat.loc[mergeHetAndSurStrat['Heterogeneity group'], 'Survival times'],
            event_observed=mergeHetAndSurStrat.loc[mergeHetAndSurStrat['Heterogeneity group'], 'Censoring values'],
            label='High heterogeneity')
    kmf.plot_survival_function(ax=ax, ci_show=True)

    results=logrank_test(durations_A=mergeHetAndSurStrat.loc[mergeHetAndSurStrat['Heterogeneity group'], 'Survival times'],
                        durations_B=mergeHetAndSurStrat.loc[~mergeHetAndSurStrat['Heterogeneity group'], 'Survival times'],
                        event_observed_A=mergeHetAndSurStrat.loc[mergeHetAndSurStrat['Heterogeneity group'], 'Censoring values'], 
                        event_observed_B=mergeHetAndSurStrat.loc[~mergeHetAndSurStrat['Heterogeneity group'], 'Censoring values'])

    print("log-rank p-value: {:.3}".format(results.p_value))

    ax.text(0.25, 0.95, f'long-rank p-value = {results.p_value:.3f}', transform=ax.transAxes, fontsize=14, color='black',verticalalignment='top')
    plt.xlabel('Months')
    plt.ylabel('Survival probability')
    plt.legend()
    plt.grid(True)
    fig.patch.set_alpha(0.0)
    plt.tight_layout()
    if saveFig:
        if n_clusters == 10:
            fig.savefig(os.path.join(figuresDir, 'survival_lowgrade_tcga.png'), bbox_inches='tight')
            fig.savefig(os.path.join(figuresDir, 'survival_lowgrade_tcga.svg'),format='svg',dpi=600,bbox_inches='tight') 
    plt.show()



def load_dataframes(n_clusters=10, uni=False):
    """Function to filter out and load all metadata for survival prediction.

    Args:
        n_clusters (int, optional): number of morphological clusters used.. Defaults to 10.
        uni (bool, optional): choice of MorphoITH endoder (if False) or UNI encoder (if True). Defaults to False.

    Returns:
        mergeHetAndSur (DataFrame): metadata with survival information for each patient.
    """

    # load the FriendlyNames mapping:
    friendlyMappingDf = pd.read_csv(files['METADATA']['tcga']['mapping'])

    # load the TCGA full signature data (which has SSIGN, TNM, etc.):
    clinDfSsign = pd.read_csv(files['METADATA']['tcga']['clinical']).set_index('patient')

    mappingDf = pd.DataFrame({'Name': [os.path.split(f)[-1].split('.')[0] for f in friendlyMappingDf['Friendly Path']],
                            'Patient': ['-'.join(os.path.split(f)[-1].split('-')[:3]) for f in friendlyMappingDf['Original Path']],
                            'Sample': ['-'.join(os.path.split(f)[-1].split('-')[:4])[:-1] for f in friendlyMappingDf['Original Path']]}).set_index('Patient')

    nameComb = pd.merge(left=mappingDf,right=clinDfSsign,left_index=True,right_index=True,how='inner')

    # load heterogeneity scores:
    if n_clusters == 10:
        finalResultsDf = pd.read_csv(os.path.join(figuresDir, 'heterogeneity_TCGA.csv'),index_col=[0]) 
        if uni:
            finalResultsDf = pd.read_csv(os.path.join(figuresDir, 'heterogeneity_TCGA_UNI.csv'),index_col=[0]) 
    else:
        finalResultsDf = pd.read_csv(os.path.join(figuresDir, 'heterogeneity_TCGA_clusters'+str(n_clusters)+'.csv'),index_col=[0]) 

    # rename survival data to more general naming:
    nameComb = nameComb.rename(columns={"PFS_MONTHS": "PFS", "ajcc_pathologic_m": "mStage",
                                        "PFS_STATUS":"PFS censoring","tumor_grade":"Grade"})
    nameComb = nameComb[['Name',"PFS","mStage","PFS censoring",'Grade']]

    mergeHetAndSur = pd.merge(left=finalResultsDf,right=nameComb,on='Name',how='inner')

    # check for blacklisted TCGA samples:
    blackListedTcga = files['METADATA']['tcga']['blacklist']
    blackListedTcgaPd = pd.read_csv(blackListedTcga)
    blackListedTcgaPd['Name'] = blackListedTcgaPd['FriendlyName'].apply(lambda x: x.split('.')[0])
    blackListedTcgaPd = blackListedTcgaPd[['Name','ReasonForBlackList']]

    mergeHetAndSur = pd.merge(mergeHetAndSur, blackListedTcgaPd,on='Name', how='left')
    mergeHetAndSur['ReasonForBlackList'] = mergeHetAndSur['ReasonForBlackList'].fillna('Not blacklisted')
    mergeHetAndSur = mergeHetAndSur[mergeHetAndSur['ReasonForBlackList']=='Not blacklisted']

    # filter rows that have no grade/survival information:
    mergeHetAndSur['PFS censoring'] = np.array([s.split(':')[0] for s in mergeHetAndSur['PFS censoring']])
    mergeHetAndSur['Grade'] = np.array([int(g[1]) if g in ['G1','G2','G3','G4'] else np.nan for g in mergeHetAndSur['Grade'].values])
    mergeHetAndSur['mStage'] = np.array([g if g in ['M0','M1'] else np.nan for g in mergeHetAndSur['mStage'].values])
    mergeHetAndSur['mStage'] = mergeHetAndSur['mStage'].replace('nan', np.nan)

    isGood_test = np.logical_and.reduce([np.isfinite(mergeHetAndSur['PFS']),
                                        np.isfinite(mergeHetAndSur['Grade']),
                                        ~mergeHetAndSur['mStage'].isna(),])

    mergeHetAndSur=mergeHetAndSur[isGood_test]
    
    mergeHetAndSur['Score'] = mergeHetAndSur['Heterogeneity']
    mergeHetAndSur['Survival times'] = mergeHetAndSur['PFS']
    mergeHetAndSur["Censoring values"] = mergeHetAndSur['PFS censoring']
    mergeHetAndSur['Censoring values'] = mergeHetAndSur['Censoring values'].astype('int')

    return mergeHetAndSur

# %%

def __main__(n_clusters=10, saveFig=False, uni=False):

    assert n_clusters in [5,10,15,20]
    
    if n_clusters == 10:
        if uni:
            mergeHetAndSur = load_dataframes(n_clusters,uni=uni)
            survival_plot(mergeHetAndSur, saveFig=saveFig, n_clusters=n_clusters)
        else:
            mergeHetAndSur = load_dataframes(n_clusters)
            het_vs_grade_mstage(mergeHetAndSur, saveFig=saveFig)
            c_index(mergeHetAndSur)
            survival_plot(mergeHetAndSur, saveFig=saveFig, n_clusters=n_clusters)

    else:
        mergeHetAndSur = load_dataframes(n_clusters)
        survival_plot(mergeHetAndSur, saveFig=saveFig, n_clusters=n_clusters)


# %%
if __name__ == "__main__":
    
    __main__()
