import logging

from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """Abstract class defining evaluation strategy for our models
    """

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        calculate the scores for the model

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Return:
            None
        """

class MSE(Evaluation):
    """Evaluation  using Mean squared Error"""

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE: {}".format(e))
            raise e
        
class R2Score(Evaluation):
    """Evaluation using R2 score"""

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating r2 score...")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2 Score:{}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating R2 score: {}".format(e))
            raise e
        
class RMSE(Evaluation):
    """Evaluating using the root mean squared Error"""

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("calculating RMSE")
            rmse = mean_squared_error(y_true, y_pred, squared= False)
            logging.info("RMSE: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in calculating RMSE: {}".format(e))
            raise e