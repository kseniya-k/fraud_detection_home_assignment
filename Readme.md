## Data preprocessing

Data is preporcessed in module `data_preparation.py` in function `prepare_data`. It consists of following steps:
1. Load data
2. Parse some date, datetime and numeric columns
3. Fill NaNs
4. Join auxiliraly data to main dataframe
5. Extract features
6. Drop unnecessary features and columns
7. Save data

I/o functions are implemented in `utils.py` module. If data can be stored in DB or other FS, just change i/o functions `load_data`, `wirte_data`.

Also, it would be useful to save some statistics of input data: mean, std, amount of NaNs in moving window; set of unique categories.
In regular train and prediction pipelines we could compare new statistics and statistincs on historical data to detect corrupted data or data drift.

## Machine Learning Methon Selection

I've chosen LightGBM Classifier model, because:

- It can naturally process categorical columns. Dataset consists most of categorical columns, that's why tree-based models can be a good approach
- It is tree-based model. Historicaly, tree-based models perform well on fraud prediction
- It can process high unbalanced data by weightening examples while training

Also, I would start with some much more simple models:

- statistical anomaly detection
- Logistic Regression models

If some cases can be detected by one of thease models with high confidence, I would try to drop such cases from train dataset for lightgbm, because they likely have different nature than others.
After that I would evaluate metrics on cross-validation to ensure that quality did not decrease.

## Validation Strategy and Metrics

I've chosen F1-score as a main metric and cross-validation as a main evaluation technique. F1-score is mean geometric from Precision and Recall, which is good for classification with unbalanced classes.
Cross-validation with 5 folds allows to evaluate model performance and avoid overfitting on one test dataset or case. As a result of `evaluate_metrics` function, meta information is saved:
- configuration
- full dump of model parameters
- model parameters that were passed to LightGBMClassifier
- F1-score on cross-validation
- If possible, learning curve is plotted

From learning curves we can derive information about under- or overfitting. Also, we can say if current amount of training steps is enough for our data and if early stopping parameters are appropriate for the data.

With little improvements, this pipeline can be tranformed into robust repeatable pipeline for experiments with model and data.

Also, I would save model loss on train and validation datasets for each training step and training time, to plot them as learning curves in separate dashboard.

## Hyperparameter Tuning

We can find optimal hyperparameters by evaluating F1-score on cross-validation, analyzing loss on train and validation, analyzing amount of trees. In my case, I tried few params:
- `is_unbalance = True` - allows lightgbm to weight objects in loss
- `sample_weight = 99` instead of `is_unbalance = True` - to restrict rate between classes (percent of class 1 (fraud transactions) in data is 0.12%) - improved f1-score, but according to learning curve there is overfitting
- `max_depth = 20` to increase overfitting - helped a bit with f1-score
- `max_depth = 10`, `early_stopping_rounds = 20` instead of 50 - improved f1-score and learning curve in terms of loss on validation dynamics.

As a result, model still have poor performance (F1-score is 0.045) and need futher improvements.

In general, we can arrange simple hyperparameter gridsearch or use tools like optuna.

## Deployment

This pipeline can be run as follows:
1. Set read data from appropriate source by changing i/o functions in `utils.py`, fix config fields `base_path`, `filename_transactions`, `filename_cards`, `filename_users` if needed (module `config.py`)
2. Change other config fields if needed
3. Run `prepare_data` function from `data_preparation.py` or create a DAG task from it or from its parts: `transform_data` and `add_features`
4. Run `evaluate_metrics` function from `train.py` to ensure that F1-score is appropriate on train data. If needed, change model paramenters in config and repeat this step
5. Run `train_produciton` function from `train.py` to train model on all available data, save model and encofing
6. Run `predict_production` function from `predict.py` to predict on new data

Trainig steps 1-5 can be done automaticaly on some schedule, for example on weekly basis.

Prediction steps 1-2, 6 can also be done automaticaly on some other schedule, for example on daily basis.


## Possible improvements
Code:
- Add unit tests to most of functions, add end-to-end tests to train and predict pipelines
- Prepare data by batches in parallel

Usability in production:
- Save data and model version after training new model. Detect latest stable model and use it for prediction.
- Detect if data for predict is new or it was already handled
- Save statistincs on train and predict data and predicts computed on moving window, alert if new data differs from old data significantly
- Also, alert if F1-score or train and validation losses differs from "usual" levels

Model performance:
- Optimize lightgbm hyperparameters, try other models
- Perform feature selection by feature importance
- Analyze few cases: take feature importance, SHAP, trees themselves, train objects related to this case
- Parse "Errors?" column
