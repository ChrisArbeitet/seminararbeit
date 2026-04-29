from . import data_cleaning_pipeline as data_cleaning_pipeline
from dataclasses import dataclass, field
from typing import Dict, List
import config
import os, glob
import pandas as pd
from . import dataset as dataset

def safe_read_excel(path, **kwargs):
    kwargs.setdefault('engine', 'calamine')
    return pd.read_excel(path, **kwargs)

class DataProcessor:
    def __init__(self, target, material, group_type):
        self.target = target
        self.material = material
        self.group_type = group_type

    def run(self):
        df_merged_machine_data, df_lab_data = self.prepare_dataset()
        df_machine_and_lab_data = pd.merge(df_merged_machine_data, df_lab_data, on='DATAKEY64', how='inner')
        df_machine_and_lab_data = self.clean_dataset(df_machine_and_lab_data)

        if 'IB_AVG' in df_machine_and_lab_data.columns:
            df_machine_and_lab_data['IB_AVG'] *= 1000
        if 'MOR_AVG' in df_machine_and_lab_data.columns:
            df_machine_and_lab_data['MOR_AVG'] *= 10

        df_mutation_rules = self.load_mutation_rules()
        df_group_rules    = self.load_group_rules()
        df_group          = self.load_groups()

        df_preprocessed_filtered = self.drop_chosen_features(df_machine_and_lab_data, df_mutation_rules, drop_type=3)
        df_preprocessed_filtered = self.sort_dataset_along_groups(df_preprocessed_filtered, df_group)
        gen_dict   = self.create_gen_dict(df_preprocessed_filtered, df_group, df_mutation_rules)
        group_dict = self.create_group_dict(df_preprocessed_filtered, df_group_rules, gen_dict)

        if df_preprocessed_filtered.columns[-1] != self.target:
            raise ValueError(f"Final Dataset has wrong target: ({df_preprocessed_filtered.columns[-1]}) instead of ({self.target})")

        final_dataset_object = dataset.Dataset(df_preprocessed_filtered, gen_dict, group_dict, self.target, self.material)
        print(f"Succesfully created dataset for target {self.target}.\n")
        return final_dataset_object

    def load_group_rules(self):
        if self.group_type == 'Version_A':
            group_rules_path = os.path.join("data/group_rules_A.xlsx")
        elif self.group_type == 'Version_B':
            group_rules_path = os.path.join("data/group_rules_B.xlsx")
        # ✅ beide Zweige nutzen jetzt safe_read_excel
        return safe_read_excel(group_rules_path, sheet_name=self.target, header=0)

    def load_groups(self):
        if self.group_type == 'Version_A':
            group_path = "data/group_Version_A.xlsx"
        elif self.group_type == 'Version_B':
            group_path = "data/group_Version_B.xlsx"
        # ✅ beide Zweige nutzen jetzt safe_read_excel
        return safe_read_excel(group_path, usecols=['name', 'group'])

    def drop_chosen_features(self, df_preprocessed, mutation_rules, drop_type):
        valid_features = mutation_rules[mutation_rules[self.target] < drop_type]['NAME']
        last_column = df_preprocessed.columns[-1]
        filtered_columns = [col for col in df_preprocessed.columns if col in valid_features.values or col == last_column]
        return df_preprocessed[filtered_columns]

    def sort_dataset_along_groups(self, df_dataset, df_group):
        last_column_name = df_dataset.columns[-1]
        last_column = df_dataset[last_column_name]
        df_dataset = df_dataset.drop(columns=[last_column_name])
        group_mapping = dict(zip(df_group['name'], df_group['group']))
        sorted_columns = sorted(df_dataset.columns, key=lambda x: (group_mapping.get(x, 'default_group'), x))
        sorted_dataset = df_dataset[sorted_columns].copy()
        sorted_dataset[last_column_name] = last_column
        return sorted_dataset

    def prepare_dataset(self):
        machine_data_files, lab_file = self.load_machine_and_lab_files()

        list_dfs_process_data = []
        for df_mft_data in machine_data_files:
            df = pd.read_csv(df_mft_data, encoding='utf-8', encoding_errors='ignore')
            if 'TRACKING_TIMESTAMP' in df.columns:
                df = df.drop(columns=['TRACKING_TIMESTAMP'])
            list_dfs_process_data.append(df)

        df_merged_machine_data = list_dfs_process_data.pop(0)
        df_merged_machine_data['DATATIMESTAMP'] = pd.to_datetime(df_merged_machine_data['DATATIMESTAMP'])
        df_merged_machine_data = df_merged_machine_data.sort_values(by='DATATIMESTAMP', ascending=True)

        column_to_rename = 'DATATIMESTAMP'
        new_columns = ['DATATIMESTAMP_{}'.format(i) for i in range(len(list_dfs_process_data))]
        for table_name, new_column in zip(list_dfs_process_data, new_columns):
            table_name.rename(columns={column_to_rename: new_column}, inplace=True)

        for df_mft_data in list_dfs_process_data:
            df_merged_machine_data = pd.merge(df_merged_machine_data, df_mft_data, on="DATAKEY64", how='inner')

        timestamp_columns = df_merged_machine_data.filter(regex='DATATIMESTAMP').columns
        df_merged_machine_data.drop(columns=timestamp_columns, inplace=True)
        df_merged_machine_data.rename(columns={
            'PRIM_KšHLWASSER_RL_TEMPERATUR':      'PRIM_KUEHLWASSER_RL_TEMPERATUR',
            'PRIM_K\\x9aHLWASSER_RL_TEMPERATUR':  'PRIM_KUEHLWASSER_RL_TEMPERATUR'
        }, inplace=True)

        df_lab_data = pd.concat([pd.read_csv(file, encoding='unicode_escape') for file in lab_file], ignore_index=True)
        df_lab_data = df_lab_data[df_lab_data['LABMATERIALNO'].isin(self.material)]
        target_column = [self.target, 'DATAKEY64']
        df_lab_data = df_lab_data[target_column]
        df_lab_data.dropna(axis=0, inplace=True)

        return df_merged_machine_data, df_lab_data

    def load_machine_and_lab_files(self):
        data_base_path = config.data_base_path
        machine_data_path = os.path.join(data_base_path, config.mft_dir)
        lab_data_path = os.path.join(data_base_path, config.lab_dir)
        machine_data_files = glob.glob(machine_data_path)
        lab_file = glob.glob(lab_data_path)
        if not lab_file:
            raise ValueError("Labordaten fehlen.")
        if not machine_data_files:
            raise ValueError("Prozessdaten fehlen.")
        return machine_data_files, lab_file

    def load_mutation_rules(self):
        mutation_rules_path = os.path.join(config.data_base_path, config.exp_knowledge)
        tags_table_path = os.path.join(config.data_base_path, config.tags_table)
        df_mutation_rules = safe_read_excel(mutation_rules_path, sheet_name='Bewertungen')
        df_tags_table = pd.read_csv(tags_table_path, sep=';', usecols=['NAME', 'LANGUAGE_1'], header=0, encoding='latin1')
        merged_df = pd.merge(df_tags_table, df_mutation_rules, on='LANGUAGE_1', how='inner')
        mutation_rules = merged_df[['NAME', self.target]]
        mutation_rules.loc[:, self.target] = pd.to_numeric(mutation_rules[self.target])
        return mutation_rules

    def clean_dataset(self, machine_and_lab_data):
        last_column_name = machine_and_lab_data.columns[-1]
        last_column = machine_and_lab_data[last_column_name]
        machine_and_lab_data = machine_and_lab_data.drop(columns=[last_column_name])
        pipeline = data_cleaning_pipeline.DataCleaningPipeline()
        pipeline.add_step(data_cleaning_pipeline.DataCleaningPipeline.delete_complex_data_type)
        pipeline.add_step(data_cleaning_pipeline.DataCleaningPipeline.drop_datakey)
        pipeline.add_step(data_cleaning_pipeline.DataCleaningPipeline.delete_duplicates)
        pipeline.add_step(data_cleaning_pipeline.DataCleaningPipeline.filter_invalid_entries)
        pipeline.add_step(data_cleaning_pipeline.DataCleaningPipeline.filter_variance)
        pipeline.add_step(data_cleaning_pipeline.DataCleaningPipeline.drop_soll_nom)
        machine_and_lab_data = pipeline.execute(machine_and_lab_data)
        machine_and_lab_data[last_column_name] = last_column
        return machine_and_lab_data

    @dataclass
    class GenInfo:
        data_proportion: float = None
        group_name: str = 'n'
        mutation_prob: float = 10
        feature_name: str = None
        memory: Dict[str, List[int]] = field(default_factory=lambda: {'DENSITY_AVG': [0, 0], 'MOR_AVG': [0, 0], 'IB_AVG': [0, 0]})

    def create_gen_dict(self, df_preprocessed, df_group, df_mutation_rules):
        gen_dict = {i: DataProcessor.GenInfo(feature_name=name_tag)
                    for i, name_tag in enumerate(df_preprocessed.columns)}
        for index in gen_dict:
            if gen_dict[index].feature_name in df_group['name'].values:
                gen_dict[index].group_name = df_group.loc[df_group['name'] == gen_dict[index].feature_name, 'group'].values[0]
            if gen_dict[index].feature_name in df_mutation_rules['NAME'].values:
                mutation_prob = df_mutation_rules.loc[df_mutation_rules['NAME'] == gen_dict[index].feature_name, self.target].values[0]
                gen_dict[index].mutation_prob = 30 - 10 * mutation_prob
        return gen_dict

    @dataclass
    class GroupInfo:
        start_index: int = None
        end_index: int = None
        min_feature: int = 0
        max_feature: int = 1000

    def create_group_dict(self, dataset_relevant_target, df_group_rules, gen_dict):
        group_names = []
        for key, _ in gen_dict.items():
            group_name = gen_dict[key].group_name
            if group_name not in group_names:
                group_names.append(group_name)
        group_dict = {group: DataProcessor.GroupInfo() for group in group_names}
        previous_group = None
        for index, gene in enumerate(dataset_relevant_target):
            current_group = gen_dict[index].group_name
            if current_group != previous_group:
                if current_group in df_group_rules['Group-Name'].values:
                    min_feature = int(df_group_rules.loc[df_group_rules['Group-Name'] == current_group, 'min_feature'].values[0])
                    max_feature = int(df_group_rules.loc[df_group_rules['Group-Name'] == current_group, 'max_feature'].values[0])
                else:
                    min_feature = 0
                    max_feature = 1000
                group_dict[current_group].start_index = index
                group_dict[current_group].end_index = None
                group_dict[current_group].min_feature = min_feature
                group_dict[current_group].max_feature = max_feature
                if previous_group is not None:
                    group_dict[previous_group].end_index = index - 1
                previous_group = current_group
        group_dict[current_group].end_index = index
        return group_dict
