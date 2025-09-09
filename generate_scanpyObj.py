import sys
import getopt

# Input files and parameteres
GEFpath = ''
registIMGfile = ''
tissueIMGfile = ''
maskIMGfile = ''
geneID = 'symbol'
prefixSpeciesArg = ''
library_id = 'stereo-seq'
description = 'sample'
imgScaleF = 0.25
grayScale = 'normal'
exprMethod = '' # GaussianMixture or LinearRegression
spotToKeep = 'underTissue'
keepSpots = True
binSize = 80
outputDir = './'


helpMessage = "generate_scanpyOnj.py arguments:\n\t-f <file.gef> (required) \n\t-r <registerdImage.tif> \n\t-t <tissueImage.tid> \n\t-m <cellmaskImage.tif> \n\t-s Low Res image scaling factor (0,...,1] default=0.25 \n\t-g greyScale ('normal' default or 'inverted') \n\t-i geneID (required, default 'symbol') \n\t-p specie prefixes (optional, default '') \n\t-l library_id \n\t-d description \n\t-b binSize (80 default) \n\t -s spotToKeep (underTissue, expressed or both) \n\t -e expressionMethod (GaussianMixture or LinearRegression) \n\t -k keepSpots \n\t-o output directory path (default './')"

try:
  opts, args = getopt.getopt(sys.argv[1:],"hf:r:t:m:c:g:i:p:l:d:s:e:b:k:o:",["GEF=","registIMG","tissueIMG","maskIMG","imgScale","greyScale","geneID","speciePrefixes","libraryID","description","spotToKeep","expressionMethod","binSize","outputDir"])
except getopt.GetoptError:
  print(helpMessage)
  sys.exit(2)
if len(opts) < 2:
    print(helpMessage)
    sys.exit(2)
for opt, arg in opts:
  if opt == '-h':
     print (helpMessage)
     sys.exit()
  elif opt in ("-f", "--GEF"):
     GEFpath = arg
  elif opt in ("-r", "--registIMG"):
     registIMGfile = arg
  elif opt in ("-t", "--tissueIMG"):
     tissueIMGfile = arg
  elif opt in ("-m", "--maskIMG"):
     maskIMGfile = arg
  elif opt in ("-c", "--imgScale"):
     imgScaleF = float(arg)
  elif opt in ("-g", "--grayScale"):
     grayScale = arg.lower()
  elif opt in ("-i", "--geneID"):
     geneID = arg
  elif opt in ("-p", "--speciePrefixes"):
     prefixSpeciesArg = arg
  elif opt in ("-l", "--libraryID"):
     library_id = arg
  elif opt in ("-d", "--description"):
     description = arg
  elif opt in ("-s", "--spotToKeep"):
     spotToKeep = arg
  elif opt in ("-e", "--expressionMethod"):
     exprMethod = arg
  elif opt in ("-b", "--binSize"):
     binSize = int(arg)
  elif opt in ("-k", "--keepSpots"):
     keepSpots = (arg.lower() == 'true')
  elif opt in ("-o", "--outputDir"):
     outputDir = arg

# Loading required packages
import os
import re
import sys

import stereo as st

import pandas as pd

from anndata import AnnData
import scanpy as sc

#import squidpy as sq

from tqdm import tqdm

from scipy import stats

import cv2

import yaml

import matplotlib.pyplot as plt

import numpy as np

from custom_functions import selectExpThr, reduceIMG


prefixSpecies = []
if (prefixSpeciesArg != ''):
    prefixSpecies = [g.strip() for g in prefixSpeciesArg.split(',')]

# Load ST data from gef
STdata = st.io.read_gef(file_path=GEFpath, 
                                  bin_type='bin',
                                  bin_size=binSize)

# adata = AnnData(STdata.exp_matrix, obsm={"spatial": STdata.position})
# adata.var_names = STdata.gene_names
# adata.obs_names = np.char.mod('%d', STdata.cell_names)

STdata.tl.raw_checkpoint()

adata = st.io.stereo_to_anndata(STdata, flavor='scanpy')


# handle ENSEMBL gene id
if geneID != 'symbol':
    ENSG_symbol = adata.var.reset_index()['index'].str.split('_', expand=True)
    adata.var['geneID'] = ENSG_symbol[0].tolist()
    adata.var['symbol'] = ENSG_symbol[1].tolist()
    adata.var.set_index('geneID', inplace=True)
    adata.var_names_make_unique()


adata.uns['binSize'] = binSize

spatial_key = "spatial"
library_id = library_id
adata.uns['description'] = description
adata.uns[spatial_key] = {library_id: {}}

tissueIMG = None
registIMG = None
maskIMG = None

if (registIMGfile != '' or tissueIMGfile != '' or maskIMGfile != ''):
    adata.uns[spatial_key][library_id]["images"] = {}
    adata.uns[spatial_key][library_id]["scalefactors"] = {"spot_diameter_fullres": int(binSize)}

    if registIMGfile != '':
        # Consider multiple files
        registIMGarr = registIMGfile.split(',')
        for imgFN in registIMGarr:

            imgTP = re.sub(r'^[\W_]+|[\W_]+$', '', imgFN.split(library_id)[0])

            registIMG = cv2.imread(imgFN, 1)#.astype("uint8")#.transpose(1, 0, 2)
            
            registIMG = (((registIMG - np.min(registIMG)) / (np.max(registIMG)-np.min(registIMG))) * 255).astype("uint8")
            
            if grayScale == 'inverted':
                registIMG = 255 - registIMG
            adata.uns[spatial_key][library_id]["images"][imgTP] = registIMG
            adata.uns[spatial_key][library_id]["scalefactors"][f"tissue_{imgTP}_scalef"] = 1
            if imgScaleF != 1:
                adata.uns[spatial_key][library_id]["images"][f"{imgTP}lowRes"] = reduceIMG(registIMG, imgScaleF)
                adata.uns[spatial_key][library_id]["scalefactors"][f"tissue_{imgTP}lowRes_scalef"] = imgScaleF

    if tissueIMGfile != '':
        tissueIMGarr = tissueIMGfile.split(',')
        tissueIMG = np.array([])
        for imgFN in tissueIMGarr:
            tissueIMGtmp = cv2.imread(imgFN, 1)#.astype("uint8")[:,:,0]#.transpose(1, 0, 2)
            tissueIMGtmp = (((tissueIMGtmp - np.min(tissueIMGtmp)) / (np.max(tissueIMGtmp)-np.min(tissueIMGtmp))) * 255).astype("uint8")[:,:,0]
            if tissueIMG.shape[0] == 0:
                tissueIMG = tissueIMGtmp
            else:
                tissueIMG += tissueIMGtmp
        
        tissueIMG[tissueIMG > 0] = 255
        adata.uns[spatial_key][library_id]["images"]["tissueMask"] = tissueIMG
        adata.uns[spatial_key][library_id]["scalefactors"]["tissue_tissueMask_scalef"] = 1
        if imgScaleF != 1:
            adata.uns[spatial_key][library_id]["images"]["tissueMasklowRes"] = reduceIMG(tissueIMG, imgScaleF)
            adata.uns[spatial_key][library_id]["scalefactors"][f"tissue_tissueMasklowRes_scalef"] = imgScaleF

    if maskIMGfile != '':
        imgIMGarr = maskIMGfile.split(',')
        for imgFN in imgIMGarr:
            imgTP = re.sub(r'^[\W_]+|[\W_]+$', '', imgFN.split(library_id)[0])

            maskIMG = cv2.imread(imgFN, 1)#.astype("uint8")[:,:,0]#.transpose(1, 0, 2)
            maskIMG = (((maskIMG - np.min(maskIMG)) / (np.max(maskIMG)-np.min(maskIMG))) * 255).astype("uint8")[:,:,0]
            
            
            maskIMG[maskIMG > 0] = 255

            adata.uns[spatial_key][library_id]["images"][f"cellMask_{imgTP}"] = maskIMG
            adata.uns[spatial_key][library_id]["scalefactors"][f"tissue_cellMask_{imgTP}_scalef"] = 1
            if imgScaleF != 1:
                adata.uns[spatial_key][library_id]["images"][f"cellMask_{imgTP}lowRes"] = reduceIMG(maskIMG, imgScaleF)
                adata.uns[spatial_key][library_id]["scalefactors"][f"tissue_cellMask_{imgTP}lowRes_scalef"] = imgScaleF

adata.uns['orig_var_names'] = [gn for gn in adata.var_names]

if prefixSpecies != None:
    for pref_i in prefixSpecies:
        adata.var[pref_i] = adata.var_names.str.startswith(pref_i)
        sc.pp.calculate_qc_metrics(adata, qc_vars=[pref_i], inplace=True)
#    for pref_i in prefixSpecies:
#        adata.var_names = [re.sub('^{0}'.format(pref_i), '', g) for g in adata.var_names]
        
if geneID != 'symbol':
    adata.var["mito"] = adata.var['symbol'].str.startswith(("mt-", "MT-"))
    adata.var["ribo"] = adata.var['symbol'].str.startswith(("Rps", "Rpl", "RPS", "RPL"))
else:
    adata.var["mito"] = adata.var_names.str.startswith(("mt-", "MT-"))
    adata.var["ribo"] = adata.var_names.str.startswith(("Rps", "Rpl", "RPS", "RPL"))

#sc.pp.calculate_qc_metrics(adata, inplace=True)

if spotToKeep == 'underTissue' or spotToKeep == 'both':
    if tissueIMG is None:
      print('ERROR: no tissue mask provided. Please check')
      sys.exit('ERROR: no tissue mask provided. Please check')
    else:
      adata.obs['underTissue'] = adata.uns[spatial_key][library_id]["images"]['tissueMask'][adata.obsm['spatial'][:,1],adata.obsm['spatial'][:,0]]
      adata.obs.underTissue[adata.obs.underTissue > 0] = 1

if (exprMethod == 'GaussianMixture' or exprMethod == 'LinearRegression') and (spotToKeep == 'expressed' or spotToKeep == 'both'):
    adata.obs['expressed'] = 0
    expThr = selectExpThr(adata.obs.total_counts, exprMethod)
    adata.obs.expressed[adata.obs.total_counts > expThr] = 1
else:
    if spotToKeep == 'expressed' or spotToKeep == 'both':
        print('WARNING: conflict between exptMethod and spotToKeep parameters. Please, check arguments passed')

if spotToKeep == 'underTissue':
    adata.obs['keep'] = adata.obs.underTissue
else:
    #if (exprMethod == 'GaussianMixture' or exprMethod == 'LinearRegression'): 
        #print('WARNING: conflict between exptMethod and spotToKeep parameters. Please, check arguments passed')
    if spotToKeep == 'expressed':
        adata.obs['keep'] = adata.obs.expressed
    else:
        adata.obs['keep'] = adata.obs.expressed + adata.obs.underTissue


if keepSpots == False:
    adata = adata[adata.obs.keep > 0, ]
    keepGene = sc.pp.filter_genes(adata, min_counts=1, inplace=False)
    adata = adata[:,keepGene[0]]


adata.write('{0}{1}_{2:d}binS.h5ad'.format(outputDir, library_id.replace(' ', '_'), binSize), compression='gzip')

