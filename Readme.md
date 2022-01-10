# Predicting protein network topology clusters from chemical structure using deep learning


Here, we present a method to predict biological functions of chemicals. The network topology analysis from "[Assessing relative bioactivity of chemical substances using quantitative molecular network topology analysis](https://pubmed.ncbi.nlm.nih.gov/22482822/)" and "[Automated QuantMap for rapid quantitative molecular network topology analysis](https://pubmed.ncbi.nlm.nih.gov/23828784/)" were extended using deep learning to predict the function of unknown chemicals. 



## Steps

In each ipynb file, provide the path for the supp_scrips
supp_script_path = '<supp_script_path>'



In each parameter_?.json file, provide necessary paths
input_file_train = train and valid set combined file path.
input_file_test = test set path

pretrained_model_path = pretrained model path for molpmofit model



## Mask the below line in the script to avoid valid and test set augmentation

'''python
valid_augmentation_list = su.get_augmentation_list(valid_df,number_of_augmentation)
number_of_augmentation_valid = valid_augmentation_list

test_augmentation_list = su.get_augmentation_list(test_df,number_of_augmentation)
number_of_augmentation_test = test_augmentation_list
            
            
number_of_augmentation_valid = number_of_augmentation
if fold == 0:
    number_of_augmentation_test = number_of_augmentation
'''



## The workflow is
With the obtained STITCH and STRING data for humans
enter into a SQL database, under different table

Give necessary input for all the .ipynb files.
database path and table names path in qmap_ppi_out.py 

The order to run scripts:
qmap_data_generation.ipynb --> provide the database path and table names in the second column of the notebook

data_preprossing_and_get_subset.ipynb
get_protein_function_for_cid.ipynb --> provide db path and download stitch protein database for humans path
    
    
Using the above output for the necessary distance threshold dataset with support greater than specified
RUN cross validation using the scripts in cross_validation folder.
For the final model run final_run with desired data input


To test desired chemicals, either "cid" or "cid name_of_the_chemical" can be provided to the script predict_new_chem.ipynb.

