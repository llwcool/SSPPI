# SSPPI_Ensemble
## data  
Stores proteins IDs  
## ensemble_data
Store the output of the penultimate fully connected layer of other models for input to the ensemble model, in the form of "npy".  
## features  
Stores various calculated protein features.  
### distance_map  
Distance matrix calculated based on protein point cloud data.  
### dssp
Protein secondary structure characteristics: DSSP.  
### hmm
HMM obtained by sequence alignment.  
### matrix
Model input obtained through generate_matrix.py.
### oridata_dismap
Protein surface concave-convex features calculated based on protein point cloud data.  
### pssm_large
PSSM obtained by sequence alignment.  
## work
Store the process results and final results of model training.  
## Usage
```python net_evalue.py best_ensemble.pdparams best_all.pdparams best_onlyseq.pdparams best_onlystr.pdparams```  
_best_ensemble.pdparams_: The parameters of ensemble model, related to __dataset.py__, __MyNet.py__.  
_best_all.pdparams_: The parameters of model whose local features constructed by two different approachrs, related to __dataset_SSPPI.py__, __MyNet_all.py__.  
_best_onlyseq.pdparams_: The parameters of model whose local features constructed only by sliding window on protein sequenct, related to __dataset_SSPPI.py__, __MyNet_onlyseq.py__.  
_best_onlystr.pdparams_: The parameters of model whose local features constructed only by spatial distance relationship, related to __dataset_SSPPI.py__, __MyNet_onlystr.py__.  
  
  In the feature selection part, we did not make any changes in the dataset, but in the model file (MyNet.py), we made selections in the input tensor of the model, so there will be different MyNets(__MyNet.py__, __MyNet_all.py__, __MyNet_onlyseq__ and __MyNet_onlystr.py__).  

If you need to retrain the model, remember to replace __MyNet.py__, __dataset.py__, __train.py__ and __net.py__. __MyNet.py__ is used to select different local features, __dataset.py__ is used to construct different model inputs, __train.py__ and __net.py__ are used to formulate different training strategies.





 
