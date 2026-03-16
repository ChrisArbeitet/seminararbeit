import pandas as pd
from sklearn.feature_selection import VarianceThreshold

class DataCleaningPipeline:
    def __init__(self):
        # List of functions to be applied to the DataFrame
        self.steps = []
    
    def add_step(self, function, **kwargs):
        """
        Adds a cleaning step to the pipeline.
        
        Parameters:
            - function (callable): A function to be applied to the DataFrame
            - kwargs: Additional parameters for the function
        """
        self.steps.append((function, kwargs))
    
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the pipeline by applying all added steps sequentially to the DataFrame.

        Parameters:
            - df (pd.DataFrame): The DataFrame to be cleaned

        Returns:
            - df (pd.DataFrame): Cleaned DataFrame
        """
        for function, kwargs in self.steps:
            df = function(df, **kwargs)
        return df
    
    @staticmethod
    def delete_complex_data_type(df_machine_and_lab_data):
        # delete columns if datatype is complex
        df_machine_and_lab_data = df_machine_and_lab_data.select_dtypes(exclude=['object'])
        return df_machine_and_lab_data
    
    @staticmethod
    def drop_datakey(df_machine_and_lab_data):
        # delete DATAKEY column
        df_machine_and_lab_data.drop(columns=['DATAKEY64'], inplace=True)
        return df_machine_and_lab_data
    
    @staticmethod
    def delete_duplicates(df_machine_and_lab_data):
        # delete all columns containing duplicate values.
        df_machine_and_lab_data = df_machine_and_lab_data.T.drop_duplicates(keep='first').T
        return df_machine_and_lab_data
    
    @staticmethod
    def filter_invalid_entries(df_machine_and_lab_data):
        """
        This function counts the number of outliers, NaN values, and zeros for each feature (column) and calculates 
        the total sum for each column. Columns where this total exceeds 30% of the total entries are deleted.

        Parameters:
            - df_machine_and_lab_data (pd.Dataframe)

        Returns: 
            - df_machine_and_lab_data (pd.Dataframe): df_machine_and_lab_data
        """
        # count invalid for each column
        total_invalid_counts = pd.Series(index=df_machine_and_lab_data.columns, dtype='int64')

        for col in df_machine_and_lab_data.columns:
            # nans and 0 per column
            invalid_count = df_machine_and_lab_data[col].isna().sum() + (df_machine_and_lab_data[col] == 0).sum()

            # calculate quantiles for IQR
            q1 = df_machine_and_lab_data[col].quantile(0.25)
            q3 = df_machine_and_lab_data[col].quantile(0.75)
            iqr = q3 - q1

            #  outlier bounds
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # count outliers
            outliers_count = ((df_machine_and_lab_data[col] < lower_bound) | (
                    df_machine_and_lab_data[col] > upper_bound)).sum()

            # sum invalides
            total_invalid_counts[col] = invalid_count + outliers_count
        # Find columns where the number of valid (non-zero and non-NaN) values is less than half of the total number
        columns_to_drop = total_invalid_counts[total_invalid_counts >= len(df_machine_and_lab_data) * 0.3].index

        df_machine_and_lab_data = df_machine_and_lab_data.drop(columns=columns_to_drop)
        return df_machine_and_lab_data

    @staticmethod
    def filter_variance(df_machine_and_lab_data, threshold=0.01):
        # deleting columns with little variety in column
        # Find columns with less than 10% unique values
        unique_counts = df_machine_and_lab_data.nunique()
        columns_to_drop = unique_counts[unique_counts < len(df_machine_and_lab_data) / 10].index

        # Drop columns with less than 10% unique values
        df_machine_and_lab_data = df_machine_and_lab_data.drop(columns=columns_to_drop)

        # we apply a threshold for variance here
        selector = VarianceThreshold(threshold=threshold)
        df_filtered = selector.fit_transform(df_machine_and_lab_data.values)
        columns = [df_machine_and_lab_data.columns[i] for i in selector.get_support(indices=True)]
        df_machine_and_lab_data = pd.DataFrame(df_filtered, columns=columns)
        return df_machine_and_lab_data
    
    @staticmethod
    def drop_soll_nom(df_machine_and_lab_data):
        # drop all columns containing SOLL or NOM in the name
        df_machine_and_lab_data = df_machine_and_lab_data.loc[:, ~df_machine_and_lab_data.columns.str.contains("SOLL")]
        df_machine_and_lab_data = df_machine_and_lab_data.loc[:, ~df_machine_and_lab_data.columns.str.contains("NOM")]
        return df_machine_and_lab_data