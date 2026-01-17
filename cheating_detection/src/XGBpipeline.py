import xgboost as xgb
import shap
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np

class XGBoostPipeline:
    def __init__(self, task="regression", params = None, test_size=0.2, random_state=42):
        self.task = task
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.explainer = None
        self.shap_values = None
        self.params = None

    # -----------------------------
    # Train/Test Split
    # -----------------------------
    def split(self, X, y):
        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )

    # -----------------------------
    # Optuna Objective
    # -----------------------------
    def _objective(self, trial, X_train, X_valid, y_train, y_valid):
        params = {
            "objective": "reg:squarederror" if self.task == "regression" else "binary:logistic",
            "eval_metric": "rmse" if self.task == "regression" else "logloss",
            "eta": trial.suggest_float("eta", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda": trial.suggest_float("lambda", 1e-3, 10.0),
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0),
        }

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        preds = model.predict(dvalid)

        if self.task == "regression":
            return mean_squared_error(y_valid, preds, squared=False)
        else:
            preds_binary = (preds > 0.5).astype(int)
            return 1 - accuracy_score(y_valid, preds_binary)

    # -----------------------------
    # Run Optuna Tuning
    # -----------------------------
    def tune(self, X_train, y_train, n_trials=30, valid_size=0.2):
        # Internal split of the provided training data
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=valid_size,
            random_state=self.random_state
        )

        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: self._objective(trial, X_tr, X_val, y_tr, y_val),
            n_trials=n_trials
        )

        self.best_params = study.best_params
        return study

    # -----------------------------
    # Train Final Model
    # -----------------------------
    def train(self, X_train, y_train, params = None):
        if params is None:
            train_params = {
            "objective": "reg:squarederror" if self.task == "regression" else "binary:logistic",
            "eval_metric": "rmse" if self.task == "regression" else "logloss",
            **self.best_params
         }
        else:
            train_params = {
            "objective": "reg:squarederror" if self.task == "regression" else "binary:logistic",
            "eval_metric": "rmse" if self.task == "regression" else "logloss",
            **params
         }

        dtrain = xgb.DMatrix(X_train, label=y_train)
        self.model = xgb.train(train_params, dtrain, num_boost_round=500)

    # -----------------------------
    # Predict
    # -----------------------------
    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    # -----------------------------
    # SHAP Analysis
    # -----------------------------
 

    def shap_analysis(self, X_sample,  plot_path = None):
        """
        Compute SHAP values and optionally display a beeswarm plot.
        """
        import matplotlib.pyplot as plt
        # Create SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)

        # Compute SHAP values
        self.shap_values = self.explainer.shap_values(X_sample)

        # Beeswarm plot
        if plot_path is None:
            return self.shap_values
        else:
             #Create and save beeswarm plot 
            plt.figure(figsize=(10, 6)) 
            shap.summary_plot(self.shap_values, X_sample, plot_type="beeswarm", show=False) 
            plt.tight_layout() 
            plt.savefig(plot_path, dpi=300) 
            plt.close()
           
            return self.shap_values





    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data.
        Supports regression and binary classification.
        """
        from sklearn.metrics import (
                            mean_squared_error,
                            r2_score,
                            accuracy_score,
                            roc_auc_score
                        )

        preds = self.predict(X_test)

        results = {}

        if self.task == "regression":
            results["rmse"] = mean_squared_error(y_test, preds, squared=False)
            results["r2"] = r2_score(y_test, preds)

        else:  # binary classification
            # Predicted probabilities
            prob = preds

            # Convert to class labels
            pred_labels = (prob > 0.5).astype(int)

            results["accuracy"] = accuracy_score(y_test, pred_labels)

            # AUC requires probability scores, not labels
            try:
                results["auc"] = roc_auc_score(y_test, prob)
            except ValueError:
                results["auc"] = None  # e.g., if only one class present

        return results
