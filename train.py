import argparse
import sys
import cms_procedures as cms
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def get_dataset(num_samples):
    """
    Returns a list of tuples of the form (procedure_attributes, procedure_success)
    """
    samples = {}  # maps procedure_id to a tuple of (attributes, outcome)
    while len(samples) < num_samples:
        #  get random procedure_id and attribrubtes
        attributes = cms.get_procedure_attributes()
        # get procedure_id
        procedure_id = attributes["procedure_id"]
        # get outcome of procedure
        outcome = cms.get_procedure_success(procedure_id)
        # add to samples
        samples[procedure_id] = (attributes, outcome)

    return samples


def split_dataset(dataset, test_split, seed):
    """
    Returns a tuple of the form (training set, test set)
    """
    # split dataset into training and test
    X = [value[0] for value in dataset.values()]
    y = [value[1] for value in dataset.values()]

    return train_test_split(X, y, test_size=test_split, random_state=seed)


def evaluate_model(model, X_test, y_test):
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {(accuracy * 100.0):.4f}%")


def train_procedure_success_model(filename, num_samples, test_split, seed):
    """
    Returns a trained model that predicts procedure success
    """
    # get the dataset
    dataset = get_dataset(num_samples)

    # split dataset into training and test sets
    X_train, X_test, y_train, y_test = split_dataset(dataset, test_split, seed)

    # train model
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    # evaluate model
    evaluate_model(model, X_test, y_test)

    # save model
    model.save_model(filename)

    return model


# trains and saves a model that predicts procedure success
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename",
        type=str,
        default="procedure_success_model.json",
        help="Model filename",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to use for training and testing",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.2,
        help="Fraction of dataset to use for testing. Must be between 0 and 1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    args = parser.parse_args()

    train_procedure_success_model(
        args.filename, args.num_samples, args.test_split, args.seed
    )
