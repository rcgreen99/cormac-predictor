import argparse
import sys
import xgboost as xgb
import cms_procedures as cms

# takes as input the trained model and a dictionary of procedure attributes and
# returns a prediction of success (True or False).
def predict_procedure_outcome(model, procedure_attributes):
    prediction_score = model.predict(procedure_attributes)
    return prediction_score[0] > 0.5


# takes as input the trained model and a dictionary of procedure attributes
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename",
        type=str,
        default="procedure_success_model.json",
        help="Model filename",
    )
    args = parser.parse_args()

    # load model
    model = xgb.load_model(args.filename)

    # get procedure attributes (could also take this as command line input
    # depending on how you want to do this)
    procedure_attributes = cms.get_procedure_attributes()

    # predict procedure outcome
    outcome = predict_procedure_outcome(model, procedure_attributes)

    # print prediction
    print(f"Procedure predicted to be succesfull: {outcome}")
