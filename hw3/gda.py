"""
Author: Nathan Miller
Institution: UC Berkeley
Date: Spring 2022
Course: CS189/289A

A Template file for CS 189 Homework 3 question 8.

Feel free to use this if you like, but you are not required to!
"""

# TODO: Import any dependencies

class GDA:
    """Perform Gaussian discriminant analysis (both LDA and QDA)."""
    def __init__(self, *args, **kwargs):
        self._fit = False

        #TODO: Possibly add new instance variables

    def evaluate(self, X, y, mode="lda"):
        """Predict and evaluate the accuracy using zero-one loss.

        Args:
            X (np.ndarray): The feature matrix shape (n, d)
            y (np.ndarray): The true labels shape (d,)

        Optional:
            mode (str): Either "lda" or "qda".

        Returns:
            float: The accuracy loss of the learner.

        Raises:
            RuntimeError: If an unknown mode is passed into the method.
        """
        #TODO: Compute predictions of trained model and calculate accuracy
        #Hint: call `predict` to simplify logic
        accuracy = None
        return accuracy

    def fit(self, X, y):
        """Train the GDA model (both LDA and QDA).

        Args:
            X (np.ndarray): The feature matrix (n, d)
            y (np.ndarray): The true labels (n, d)
        """
        #TODO: Train both the QDA and LDA model params based on the training data passed in
        # This will most likely involve setting instance variables that can be accessed at test time
        self._fit = True

    def predict(self, X, mode="lda"):
        """Use the fitted model to make predictions.

        Args:
            X (np.ndarray): The feature matrix of shape (n, d)

        Optional:
            mode (str): Either "lda" or "qda".

        Returns:
            np.ndarray: The array of predictions of shape (n,)

        Raises:
            RuntimeError: If an unknown mode is passed into the method.
            RuntimeError: If called before model is trained
        """
        if not self._fit:
            raise RuntimeError("Cannot predict for a model before `fit` is called")

        preds = None
        if mode == "lda":
            #TODO: Compute test-time preditions for LDA model trained in 'fit'
            preds = None
        elif mode == "qda":
            preds = None
        else:
            raise RuntimeError("Unknown mode!")
        return preds
