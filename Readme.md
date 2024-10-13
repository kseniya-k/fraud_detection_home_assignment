## Data preprocessing

Data is preporcessed in module `data_preparation.py`:
1. Load data
2. Parse some date, datetime and numeric columns
3. Fill NaNs
4. Join auxiliraly data to main dataframe
5. Extract features
6. Drop unnecessary features and columns 
7. Save data

I/o functions are in `utils.py` module. If data can be stored in DB or other FS, just change i/o functions `read_data`, `wirte_data`.

Also, it would be useful to save some statistics of input data: mean, std, amount of NaNs in moving window; set of unique categories. 
When there will be a regular train and prediction pipelines, we would compare new statistics and existics to detect currepted data or data drift.

## Machine Learning Methon Selection

I've chosen LightGBM Classifier model, because:

- It can naturally process categorical columns. Dataset consists most of categorical columns, that's why tree-based models can be a good approach
- It is tree-based model. Historicaly, tree-based models perform well on fraud prediction
- It can process high unbalanced data by weightening examples while training 

Also, I would start with some much more simple models:

- statistical anomaly detection
- Logistic Refression models (rememner: all columns should be equaly scaled!)
  
If some cases can be detected by one of these models with high confidence, I would try to drop this cases from train dataset for lightgbm because such cases likely have different nature than others.