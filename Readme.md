# [Predicting protein network topology clusters from chemical structure using deep learning](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-022-00622-7)
Here, we present a method to predict the clusters that new chemicals belong to based on network topology. The network topology analyses from "[Assessing relative bioactivity of chemical substances using quantitative molecular network topology analysis](https://pubmed.ncbi.nlm.nih.gov/22482822/)" and "[Automated QuantMap for rapid quantitative molecular network topology analysis](https://pubmed.ncbi.nlm.nih.gov/23828784/)" were extended using deep learning to enable the assignment of new/unknown chemicals to predefined clusters. 
## Steps  
<br>

---

### Setting up the environment
To create and activate the environment. <br>
```bash
conda env create -f environment.yaml
conda activate qmpred
```
To export the conda environment to jupyter notebook. <br>
```bash
python -m ipykernel install --user --name=qmpred
```
<br>

---

### Data preprocessing
The data collection and preprocessing of the data are done using [.ipynb files in preprocessing_scripts](preprocessing_scripts). <br>
1. [1_create_stitch_string_sql_db.ipynb](preprocessing_scripts/1_create_stitch_string_sql_db.ipynb)  <br>
    * Download and convert the necessary data from STITCH and STRING to an sql database. <br><br>
2. [2_data_generation.ipynb](preprocessing_scripts/2_data_generation.ipynb)  <br> 
    * Quantmap is run using the interaction data from the databases. The data are then assigned to clusters based on their similarity using K-Mean clustering based on a range of distance parameters.<br><br>
3. [3_data_preprocessing.ipynb](preprocessing_scripts/3_data_preprocessing.ipynb)  <br> 
    * From all the clusters obtained from the above step those clusters with low support are rejected. <br><br>
4. [4_get_protein_function_of_clusters.ipynb](preprocessing_scripts/4_get_protein_function_of_clusters.ipynb)  <br> 
    * For the clusters selected above chemical-protein information from STITCH is used to determine the main functions of proteins in each cluster. <br><br>
6. [5_data_splits.ipynb ](preprocessing_scripts/5_data_splits.ipynb )  <br>
    * Split the dataset for cross validation and final training of the model. <br><br>
---

### Evaluation
Initially different architectures were evaluated using cross validation based on a subset of data. The architectures explored are present in the directory [cross_validation](cross_validation). The parameters for the architectures can be passed using their respective json file (the parameters given here are the default values). <br><br>

---

### Training
For the final training of the [MolPMoFiT](final_run/molpmofit_run.ipynb) architecture, the entire dataset is used. The parameters can be passed using [parameters.json](final_run/parameters.json) file. In order to run the final training of the MolPMoFiT model, pretraining has to be first carried out using 
[pretraining_molpmofit.ipynb](cross_validation/molpmofit/pretraining_molpmofit.ipynb). After the training of the final model it can be used to make predictions for new chemicals [predict_new_chem.ipynb](final_run/predict_new_chem.ipynb). The input for the prediction can be given in the text file "[test_cids.txt](final_run/test_cids.txt)" with CIDs as input. <br><br>

---

## Citation
  
Please cite:
>Predicting protein network topology clusters from chemical structure using deep learning.<br>
>Akshai Parakkal Sreenivasan, Philip J Harrison, Wesley Schaal, Damian J Matuszewski, Kim Kultima, and Ola Spjuth.<br>
>Status: Published.
