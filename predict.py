import logging

from config import Config
from data_preparation import prepare_data
from train import apply_encoding
from utils import load_data, load_encoding, load_model, write_data


def predict_production(config: Config, output_name: str):
    """
    Predict in production: prepare data and encode, load model, predict, save predict

    ToDo: check if this data was already handled
    ToDo: save logs with predicted ids: user - card - transaction datetime
    ToDo: scan directory for new data
    ToDo: add logs
    """
    logging.info("Load and prepare data")
    prepare_data(config, output_name="data_prepared_test.csv")
    # drop `target` columns for testing; in real conditions, there will be to target column
    data = load_data(config, "data_prepared_test.csv").drop(columns=["target"])

    logging.info("Load encoding and model")
    encoding = load_encoding(config, config.encoding_filename)
    model = load_model(config, config.model_filename)

    X_test_encoded = apply_encoding(data, config.categorical_columns, encoding)

    logging.info("Predict and save")
    predict = model.predict(X_test_encoded)
    data["predict"] = predict
    write_data(config, data, output_name)


# how to test:
# config = Config()
# predict_production(config, output_name="predict.csv")
