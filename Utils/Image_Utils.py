# %%

import re
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

# %%

def GetXmlAnnoNames(xmlFile):
    tree = ET.parse(xmlFile)
    root = tree.getroot()
    annoNames=[]
    subAnnoNames=[]  
    
    for annoNum,anno in enumerate(root.iter('Annotation')):
        annoNames.append(anno.get('Name'))
        regionList = anno.find('Regions')
        regionNames=[]

        for item in regionList:
            regionName=item.get('Text')
            if(regionName != None):
                regionNames.append(item.get('Text'))   
            
        subAnnoNames.append(regionNames)        
    return annoNames,subAnnoNames

def ReadRegionsFromXML(xmlFile,layerNum=0):
    tree = ET.parse(xmlFile)
    root = tree.getroot()

    regionPos = []
    regionNames = []
    isNegative=[]
    regionInfo=[]
    
    for annoNum,anno in enumerate(root.iter('Annotation')):
        if(annoNum==layerNum):
            regionList = anno.find('Regions')
            for item in regionList:
                regionName=item.get('Text')
                if(regionName != None):
                    regionNames.append(item.get('Text'))
                    isNegative.append(item.get('NegativeROA')=='1')
                    idNum=item.get('Id')
                    inputRegionId=item.get('InputRegionId')
                    vertexList = item.find('Vertices')
                    
                    numberOfVertices = sum(1 for i in vertexList)
                    pos = np.zeros((numberOfVertices, 2))
                    counter = 0
                    for v in vertexList:
                        pos[counter, 0] = int(float(v.get('X')))
                        pos[counter, 1] = int(float(v.get('Y')))
                        counter = counter + 1
                    regionPos.append(pos)
                    info={'Length':float(item.get('Length')),'Area':float(item.get('Area')), 
                          'BoundingBox':np.array([np.min(pos[:,0]),np.min(pos[:,1]),np.max(pos[:,0]),np.max(pos[:,1])]),
                          'id':idNum,'inputRegionId':inputRegionId,'Type':float(item.get('Type'))}
                    regionInfo.append(info)
    regionPos = regionPos
    isNegative=np.array(isNegative)
    return regionPos,regionNames,regionInfo,isNegative

def GetLoops(pos):
  rowCounter=0
  continueSearch=True
  loops=[]
  while continueSearch:
    idx=np.where(np.logical_and(pos[(rowCounter+1):,0]==pos[rowCounter,0], 
                                    pos[(rowCounter+1):,1]==pos[rowCounter,1]))[0]
    if(len(idx)>0):
      newPos=idx[0]+rowCounter+1
      
      loops.append(pos[rowCounter:min(newPos+1,len(pos)),:])
      rowCounter=newPos+1
      if(rowCounter>=len(pos)):
        continueSearch=False
    else:
      continueSearch=False  
      loops.append(pos[rowCounter:,:])
  return loops       

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def GetQPathTextAnno(annoFile, checkForDisjointAnnos=True):
  with open(annoFile) as f:
      content = f.readlines()
  
  regionPos = []
  regionNames = []
  isNegative=[]
  regionInfo=[]

  for p in range(len(content)):
    line=content[p]  
    className,data=line.replace(']','').split('[')
    if(isinstance(className,str) and len(className)>0 and (className[0]=='+' or className[0]=='-')):
      className=className  
    else:
      className='+'+className
    regionNames.append(className)
    
    data=np.array([float(x) for x in  data.replace('Point: ','').split(',')])
    coords=np.zeros((int(len(data)/2),2))
    coords[:,0]=data[0::2]
    coords[:,1]=data[1::2]
     
    if(checkForDisjointAnnos):
      loops=GetLoops(coords)
      loopAreas=[PolyArea(l[:,0],l[:,1]) for l in loops]
      coords=loops[np.argmax(loopAreas)]
    
    regionPos.append(coords)

    bbox=np.array([np.min(coords[:,0]),np.min(coords[:,1]),np.max(coords[:,0]),np.max(coords[:,1])])
    l=((bbox[2]-bbox[0])+1)
    w=((bbox[3]-bbox[1])+1)
    area=l*w
    info={'Length':max(l,w),'Area':area,'BoundingBox':bbox,'id':p,'inputRegionId':0,'Type':0}
    regionInfo.append(info)
    
  regionPos=regionPos
  isNegative=np.zeros(len(content))==1
  return regionPos,regionNames,regionInfo,isNegative

def MaskFromXML(annoFile,layerName,slideDim,downSampleFactor=1,distinguishAnnosInClass=True,
                outerBoxLabel='square',outerBoxClassName='BG'):  
 
  outputDim=tuple(np.int32(np.array(slideDim)/downSampleFactor))
  if(annoFile.endswith('.xml')):
    annoNames,subAnno=GetXmlAnnoNames(annoFile)
    layersToUse=[i for i,x in enumerate(annoNames) if x==layerName]
  else:
    layersToUse=[0]
    
  for annoLayer in layersToUse:
   
    if(annoFile.endswith('.xml')):
      annos=ReadRegionsFromXML(annoFile,layerNum=annoLayer)
      regionPos=annos[0]
      regionNames=annos[1]

    else:
      regionPos,regionNames,regionInfo,isNegative=GetQPathTextAnno(annoFile)
    regionNamesUpdated = []
    for reg in regionNames:
        regionNamesUpdated.append('+'+reg)
    
    outerBoxIdx=np.where([(bool(re.compile(r"^[+-]\s*").match(i)) and (i == '+'+outerBoxLabel)) for 
                                       i in regionNamesUpdated])[0]
    validNameIdx=np.where([(bool(re.compile(r"^[+-]\s*").match(i)) and not(i == '+'+outerBoxLabel)) for 
                                       i in regionNamesUpdated])[0]
    validNames=np.array(regionNamesUpdated)[:]                    
    validIsPos=[s[0]=="+" for s in validNames]
    validSuffixes=[s[1:] for s in validNames] 
    
    if(distinguishAnnosInClass):
      numberOfAnnos=np.sum(validIsPos)
    else:
      numberOfAnnos=len(np.unique(validSuffixes))
      nameToNum={}
      for num,name in enumerate(np.unique(validSuffixes)):
        nameToNum[name]=num+1
        
    maskToClassDict={}
    if(numberOfAnnos<254):
       mask = Image.new("P",outputDim, 0)
    else:
       mask = Image.new("I",outputDim, 0)
    
    regionOrder=np.concatenate([validNameIdx[np.where(validIsPos)[0]],validNameIdx[np.where(np.logical_not(validIsPos))[0]]])
    validSuffixes=np.array(validSuffixes)
    validIsPos=np.array(validIsPos)
    namesOrdered=np.concatenate([validSuffixes[np.where(validIsPos)[0]],validSuffixes[np.where(np.logical_not(validIsPos))[0]]])
    isPosOrdered=np.concatenate([validIsPos[np.where(validIsPos)[0]],validIsPos[np.where(np.logical_not(validIsPos))[0]]])
    
    for regionCounter,regionIdx in enumerate(regionOrder):
      
      pos=regionPos[regionIdx]/downSampleFactor
      name=namesOrdered[regionCounter]
      isPos=isPosOrdered[regionCounter]
      
      poly=np.array(pos)
      poly=np.concatenate((poly,np.expand_dims(poly[0,:],axis=0)))   
      if isPos:
        if(distinguishAnnosInClass):
            annoClass=regionCounter+1
        else:
            annoClass=nameToNum[name]
        maskToClassDict[annoClass]=name
        ImageDraw.Draw(mask).polygon(poly.ravel().tolist(), outline=int(annoClass), fill=int(annoClass))
      else: 
        ImageDraw.Draw(mask).polygon(poly.ravel().tolist(), outline=0, fill=0)
    mask=np.array(mask)
    
    if(len(outerBoxIdx)>0):
      if(numberOfAnnos<254):
        outerMask = Image.new("P",outputDim, 0)
      else:
        outerMask = Image.new("I",outputDim, 0)
      for oIdx in outerBoxIdx:
        pos=regionPos[oIdx]/downSampleFactor
        poly=np.array(pos)
        poly=np.concatenate((poly,np.expand_dims(poly[0,:],axis=0)))   
        ImageDraw.Draw(outerMask).polygon(poly.ravel().tolist(), outline=int(numberOfAnnos+1), fill=int(numberOfAnnos+1))
      outerMask=np.array(outerMask)
      mask[np.logical_not(outerMask>0)]=0
      mask[np.logical_and(outerMask>0,mask==0)]=numberOfAnnos+1
      maskToClassDict[numberOfAnnos+1]= outerBoxClassName
    
    return mask,maskToClassDict
