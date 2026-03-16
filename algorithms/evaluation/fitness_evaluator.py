import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from algorithms.evaluation.linear_regression import Regressor

"""
Module provides FitnessEvaluator class to evaluate the fitness of an individual in the feature selection process.
"""

class FitnessEvaluator:
    """
    Class provides an evaluate method to evluate an individual by splitting the input data and evaluating the 
    fitness based on the results of the regression model

    Parameters:
        - regressor (Regressor): Object to train the model with by performing regression
        - data_split (float): Proportion of the data used for training (the rest is used for testing)
        - dataset (pd.Dataframe): Dataframe with the input data for each features
        - max_features (int): Maximum number of features that should be activated
        - caching (bool): If True, caching is enabled
        - penalty (bool): If True, a penalty is applied for too many activated features
    
    """
    def __init__(self, target, data_split, dataset, max_features, caching=True, penalty=True):
        self.regressor = Regressor(target_column=target)
        self.data_split = data_split
        self.max_features = max_features
        self.dataset = dataset 
        self.caching = caching
        self.penalty = penalty
        self.scores_cache = {} if caching else None

    def evaluate(self, individual):
        """
        Evaluates the fitness of an individual by splitting the data of the dataset into test- and traingsdata 
        and training a regression model with it

        Parameters:
            - individual (list[deap.creator.Individual]): Binary list of features to be activated. The list can also be list[int]

        Returns:
            - score (list[float]): The fitness value of the individual. Length of the list is 1
        """
        individual_tuple = tuple(individual)

        # Check if score of individual is already in cache
        if self.caching and individual_tuple in self.scores_cache:
            return self.scores_cache[individual_tuple]

        # Set last column of individual to 1 because it is the target column
        individual[-1] = 1


        # Reduce the dataset to the activated features
        dataset_genome = self.dataset.loc[:, np.array(individual, dtype=bool)]

        # Remove rows with NaN-values
        dataset_genome = dataset_genome.dropna(axis=0)

        # Split the data into test and training
        score = self._split_and_evaluate(individual, dataset_genome)
        
        # Caching, if activated
        if self.caching:
            self.scores_cache[individual_tuple] = (score,)

        return (score,)

    def _split_and_evaluate(self, individual, dataset_genome):
        """
        Splits the dataset and evaluates the fitness of the individual based on the regression model

        Parameters:
            - individual (list[deap.creator.Individual]): Binary list of features to be activated. The list can also be list[int]
            - dataset_genome (pd.Dataframe): Dataframe reduced to the activated features

        Returns:
            - score (float): The fitness value of the individual
        """
        try:
            train_data, test_data = train_test_split(dataset_genome, train_size=self.data_split, random_state=11)
        except Exception as e:
            print(f"Error during train-test split: {e}")
            return 100000

        if train_data.shape[0] <= 1 or test_data.shape[0] <= 1:
            print("Less than 2 datapoints left after splitting.")
            return 100000

        if train_data.shape[0] < 50:
            print(f"Less than 50 datapoints left, only {train_data.shape[0]} left.")
            return 100000

        if train_data.shape[1] <= 1:
            print(f"No feature left in training set, only {train_data.shape[1]} features.")
            return 100000

        # Split X and Y for regression model
        y_train = pd.DataFrame(train_data.iloc[:, -1])
        x_train_selected = train_data.drop(train_data.columns[-1], axis=1)
        x_test_selected = test_data.drop(test_data.columns[-1], axis=1)
        y_test = pd.DataFrame(test_data.iloc[:, -1])

        # Train and evaluate the model
        self.regressor.x_train = x_train_selected
        self.regressor.x_test = x_test_selected
        self.regressor.y_train = y_train
        self.regressor.y_test = y_test

        if config.regression == ['OLS']:
            model, model_predictions = self.regressor.perform_ols()

        if config.regression == ['PLS']:
            model, model_predictions = self.regressor.perform_pls()

        if config.regression == ['Ridge']:
            model, model_predictions = self.regressor.perform_ridge()

        # Calculate the fitness
        return self._calculate_fitness(individual, self.regressor, model, model_predictions, y_test)

    def _calculate_fitness(self, individual, regressor, model, model_predictions, y_test):
        """       
        Calculates the fitness value based on the model predictions and the number of selected features

        Parameters:
            - individual (list[deap.creator.Individual]): Binary list of features to be activated. The list can also be list[int]
            - regressor (Regressor): Object to train the model with by performing regression
            - model (object): Trained model
            - model_predictions (np.array): Predictions of the model 
            - y_test (pd.Dataframe): Test data of the target

        Returns:
            score (float): A score based on the accuracy of the model
        """
        hard_constraint = False
        scores = regressor.regression_evaluation(actual_values=y_test.values, predicted_values=model_predictions, model=model)
        criteria_weights = np.array([1, 1, 1, 1, -1, 1])
        scores = scores.values
        model_scores_weighted = np.multiply(scores, criteria_weights)
        score = sum(model_scores_weighted)

        # Penalty for to many acitvated features
        individual_sum = np.sum(individual, axis=0)
        if individual_sum > self.max_features and self.penalty:
            diff_features = abs(int(individual_sum - self.max_features))
            if diff_features != 0:
                if hard_constraint:
                    score = float('inf')
                else:
                    score = (abs(score) + 0.7 * diff_features) * (1.5 ** diff_features)

        return score