import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract class for all models
    """
    @abstractmethod
    def train(self, X_train, y_train):
        """Trains the Model
        Args: 
            X_train: Training data
            y_train: Training labels

        Returns:
            None
        """
        pass

class LinearRegressionModel(Model):
    """Linear Regression Model"""

    def train(self,X_train, y_train, **kwargs):
        """Trains a  linear model
        Args:
            X_train: training data
            y_train: training labels

        returns:
            None
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model training completed")
            return reg
        except Exception as e:
            logging.error("Error in training Model: {}".format(e))
            raise e
        