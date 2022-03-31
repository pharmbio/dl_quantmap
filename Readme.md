# Predicting protein network topology clusters from chemical structure using deep learning


Here, we present a method to predict biological functions of chemicals. The network topology analysis from "[Assessing relative bioactivity of chemical substances using quantitative molecular network topology analysis](https://pubmed.ncbi.nlm.nih.gov/22482822/)" and "[Automated QuantMap for rapid quantitative molecular network topology analysis](https://pubmed.ncbi.nlm.nih.gov/23828784/)" were extended using deep learning to predict the function of unknown chemicals. 



## Steps  
  
### Data preprocessing

The data collection and preprocessing of the data is done using .ipynb files in preprocessing_scripts. <br>
The scrips are run in order. <br>
1. [1_create_stitch_string_sql_db.ipynb](preprocessing_scripts/1_create_stitch_string_sql_db.ipynb)  <br>
    * Download and convert necessary data from STITCH and STRING to sql database. <br><br>
2. [2_data_generation.ipynb](preprocessing_scripts/2_data_generation.ipynb)  <br> 
    * From the interaction data from the above step. Quantmap is ran for the dataset. The data is then converted to clusters based on their similarity using K-Mean clustering using different distance parameters.<br><br>
3. [3_data_preprocessing.ipynb](preprocessing_scripts/3_data_preprocessing.ipynb)  <br> 
    * Clusters are based on support per clusters. Clusters with lower support are rejected. <br><br>
4. [4_get_protein_function_of_clusters.ipynb](preprocessing_scripts/4_get_protein_function_of_clusters.ipynb)  <br> 
    * For the obtained clusters from above, function of proteins in the clusters were obtained and assigned as the function of the cluster. <br><br>
6. [5_data_splits.ipynb ](preprocessing_scripts/5_data_splits.ipynb )  <br>
    * Split the dataset for cross validation and final training of the model. <br><br>


### Evaluation

Initially different architectures are evaluated using cross validation using a subset of data. Different architectures are present in the directory cross_validation. The parameters for the architectures can be passed using respective json file. <br><br>

### Training

For the final training of the MolPMoFiT architecture the entire dataset is used. The parameters can be passed using [parameters.json](final_run/parameters.json) file. In order to run the final training of the MolPMoFiT, pretraining of the architecture has to be carried out using 
[pretraining_molpmofit.ipynb](cross_validation/molpmofit/pretraining_molpmofit.ipynb). Post training, the model can be used to predict using [predict_new_chem.ipynb](final_run/predict_new_chem.ipynb). The input for the prediction can be given in the text file "[test_cids.txt](final_run/test_cids.txt)" as CIDs as input. <br><br>


  
## Citation
  
Please cite: Predicting protein network topology clusters from chemical structure using deep learning  
Status: Submitted  
