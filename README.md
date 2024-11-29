# Replication Package for *Decoding the tackle: Using a Machine Learning approach to understand direct head contact events in Elite Women's Rugby*

## Repository contents
* Pre-processing code can be found in `pre_process.ipynb`. 
* Mutual information calculations and graphs are located in `mutual_information.ipynb`
* Code for model selection and grid-search can be found in `grid_search.py`.
  * The logs from experiments are available in `logs`.
  * Models are saved in the `models` directory.
  * Helper functions are available in `utils.py`
  * Random seeds used are stored in `random_seeds.csv` and the code to generate them is given in `gen_seeds.py`.
* Evaluation is performed in `evaluation.ipynb`.
* `images` contains all generated figures.
* `*_mi.csv` files contain mutual information analysis for each target variable.
* `frequency_label.csv` contains value counts for each variable in dataset. 

## To use custom data: 
In order to use custom data, the excel sheet will need to be placed into a folder named `data`. Variables in Excel format should follow the same variable names and number of levels. For this information, please see `pre_process.ipynb`. Data with variable names and levels different from those used in this format are not supported by default. 
