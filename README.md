# Procedure Success Predictor

## Model Selection

A decision tree was used for the model since the problem does not involve computer vision or natural language proccessing (which would be better addressed with deep learning). Additionally, the problem is in the healthcare domain, which frequently requires model prediciton explainability, at which decisions tree excel.

## Notes

1. Whether the attributes returned by the `get_procedure_attributes()` function are consistent across calls (i.e. they always return the same attributes keys) or they are not consistent arcorss calls should not matter, since, XGBoost handles missing datas by default.

2. A sinlge model architecture is used (it does not investigate the procedure atttirbutes returned by `get_procedure_attributes()` function to differentiate model architecture) because of time constrints.

3. Did not use the `get_procedure_outcomes()` function because that would have leaked data into the model and those values are not available at inference time anyway (i.e. outcomes is not a part of the input for `predict_procedure_outcome()`).

## Training and Prediction

1. Install requirments using the command `pip3 install -r requirements.txt`.

2. Train and save the model by running the `train.py` script with desired args.

3. Use the model for predictions by running the `predict.py` script.
