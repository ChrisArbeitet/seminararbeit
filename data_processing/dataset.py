import pandas as pd

class Dataset:
    """
    This class is used as the final Dataset-object
    """
    def __init__(self, df_preprocessed, gen_dict, group_dict, target, material):
        self.df_preprocessed = df_preprocessed
        self.gen_dict = gen_dict
        self.group_dict = group_dict
        self.target = target
        self.max_features = self.get_max_features(target)
        self.info_print = None
        self.material = material
        self.violations = {generation: {i: 0 for i in range(11)} for generation in range(1, 305)}

    def save_preprocessed_dataset(self, preprocessed_dataset_path):
        """"
        Optional: Saves the final Dataset to an excel-file
        """
        self.df_preprocessed.to_excel(preprocessed_dataset_path, index=False)
        print(f"Preprocessed datset for target {self.target} saved.")

    def get_max_features(self, target):
        """
        Returns:
            - max_features (int): Maximum number of features that should be acitvated
        NOTE: If the target is not in the dictionary, it defaults to 20.
        """
        max_features = {
            'DENSITY_AVG': 16,
            'MOR_AVG': 20,
            'IB_AVG': 20,
        }
        return max_features.get(target, 20)

    def get_crossover_points(self):
        """
        Returns:
            - crossover_points_as_list (list[int]): List of indices where a new group starts in the dataset, relevant for Group-Crossover
        """
        crossover_points_as_list = [0]
        current_group = None
        for index, _ in enumerate(self.df_preprocessed.columns):
            paramter_group = self.gen_dict[index].group_name
            if paramter_group != current_group:
                if index != 0:
                    crossover_points_as_list.append(index)
                current_group = paramter_group
        return crossover_points_as_list
    
    def save_group_dict_to_excel(self, filename="group_dict_overview.xlsx"):
        """
        Creates an excel to check the correctness of the Group-Dictionary
        """
        # Convert the dictionary to a DataFrame
        data = {
            "Group Name": list(self.group_dict.keys()),
            "Start Index": [info.start_index for info in self.group_dict.values()],
            "End Index": [info.end_index for info in self.group_dict.values()],
            "Min Feature": [info.min_feature for info in self.group_dict.values()],
            "Max Feature": [info.max_feature for info in self.group_dict.values()],
        }
        df = pd.DataFrame(data)
        
        # Save the DataFrame to an Excel file
        df.to_excel(filename, index=False)
        print(f"Group dictionary saved to {filename}")
