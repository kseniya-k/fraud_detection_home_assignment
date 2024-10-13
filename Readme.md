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

## Validation Strategy and Metrics

I've chosen F1-score as a main metric and cross-validation as a main evaluation technique.
Also, I would save model loss on train and validation datasets for each training step and training time.
Loss values can be plotted as learning curves, and from learning curves we can derive information about under- or overfitting. Also, we can say if current amount of training steps is enough for our data and if early stopping parameters are appropriate for the data.

F1-score is mean geometric from Precision and Recall, which is good for classification with unbalanced classes.

Cross-validation with 5 folds allows to evaludate model and avoid overfitting on one test dataset or case.

## Hyperparameter Tuning

## Deployment

This pipeline can be deployed as follows:
1. Set read data from appropriate source by changing i/o functions in `utils.py`, fix config fields `base_path`, `filename_transactions`, `filename_cards`, `filename_users` if needed (module `config.py`)
2. Change other config fields if needed
3. Run `prepare_data` function from `data_preparation.py` or create a DAG task from it or from its parts: `transform_data` and `add_features`
4. Run `evaluate_metrics` function from `train.py` to ensure that F1-score is appropriate on train data
5.
