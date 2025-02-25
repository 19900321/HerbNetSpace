# recall_calculator.py
import pandas as pd

class DiseaseDrugRecall:
    def __init__(self, prob_filepath, true_filepath):
        self.prob_filepath = prob_filepath
        self.true_filepath = true_filepath
        self.true_disease_drug_dict = {}
        self.prob_disease_drug_dict = {}
        self.merge_data = pd.DataFrame()
        self.load_data()
    
    def load_data(self):
        try:
            df_prob = self.prob_filepath  # 直接使用传入的 DataFrame
            df = pd.read_csv(self.true_filepath, sep='\t').iloc[:, [0, 1, 2]]

            # 确保 id 列在两个 DataFrame 中是相同的类型
            df_prob['id'] = df_prob['id'].astype(str)
            df['id'] = df['id'].astype(str)
            self.merge_data = pd.merge(df_prob, df, on='id', how='left')
            self.process_data()
        except Exception as e:
            print(f"An error occurred: {e}")

    def process_data(self):
        true_df = self.merge_data.iloc[:, [1, 3, 4]]
        true_df = true_df[true_df['label'] == 1]
        grouped_true = true_df.groupby('disease')
        unique_true = grouped_true['drug'].apply(lambda x: sorted(x.unique())).reset_index()
        handle_df = pd.DataFrame(unique_true)
        
        prob_df = self.merge_data.iloc[:, [0, 3, 4]]
        df_sorted = prob_df.groupby('disease', group_keys=False).apply(lambda x: x.sort_values('prob', ascending=False))
        df_sorted['prob_id'] = df_sorted['prob'].apply(lambda x: 1 if x >= 0.5 else 0)
        filtered_df = df_sorted[df_sorted['prob_id'] == 1]
        grouped_prob = filtered_df.groupby('disease')
        unique_prob = grouped_prob['drug'].apply(lambda x: sorted(x.unique())).reset_index()
        unique_prob = pd.DataFrame(unique_prob)
        
        intersection = list(set(unique_prob['disease']) & set(handle_df['disease']))
        filter_true = handle_df[handle_df['disease'].isin(intersection)]
        filter_prob = unique_prob[unique_prob['disease'].isin(intersection)]

        for index, row in filter_true.iterrows():
            disease = row['disease']
            drugs_list = row['drug']
            self.true_disease_drug_dict[disease] = drugs_list

        for index, row in filter_prob.iterrows():
            disease = row['disease']
            drugs_list = row['drug']
            self.prob_disease_drug_dict[disease] = drugs_list
        print(self.true_disease_drug_dict)
        

    def recall(self, true_associations, predicted_associations):
        recall_scores = []
        for disease, true_drugs in true_associations.items():
            if disease in predicted_associations:
                predicted_drugs = predicted_associations[disease]
                if len(true_drugs) > 0:
                    true_positives = set(predicted_drugs) & set(true_drugs)
                    recall = len(true_positives) / len(true_drugs)
                    recall_scores.append(recall)
        return sum(recall_scores) / len(recall_scores) if recall_scores else 0

    def calculate_recall_at_k(self, predicted_associations, k):
        recall_scores = []
        for disease in self.true_disease_drug_dict:
            true_set = set(self.true_disease_drug_dict[disease])
            predicted_list = predicted_associations[disease][:k]
            predicted_set = set(predicted_list)
            recall = len(true_set & predicted_set) / len(true_set)
            recall_scores.append(recall)
        return sum(recall_scores) / len(recall_scores)

    def precision_at_k(self, predicted_associations, k):
        precision_scores = []
        for disease, true_drugs in self.true_disease_drug_dict.items():
            if disease in predicted_associations:
                predicted_drugs = predicted_associations[disease][:k]
                true_positives = set(predicted_drugs) & set(true_drugs)
                precision = len(true_positives) / k
                precision_scores.append(precision)
        return sum(precision_scores) / len(precision_scores)
    
    def f1_at_k(self, predicted_associations, k):
        recall_at_k = self.calculate_recall_at_k(predicted_associations, k)
        precision_at_k = self.precision_at_k(predicted_associations, k)
        if recall_at_k + precision_at_k == 0:
            return 0
        return 2 * (recall_at_k * precision_at_k) / (recall_at_k + precision_at_k)
    
    def calculate_metrics(self, k):
        average_recall_scores = self.recall(self.true_disease_drug_dict, self.prob_disease_drug_dict)
        average_recall_at_k = self.calculate_recall_at_k(self.prob_disease_drug_dict, k)
        precision_scores_at_k = self.precision_at_k(self.prob_disease_drug_dict, k)
        f1_scores_at_k = self.f1_at_k(self.prob_disease_drug_dict, k)
        return average_recall_scores, average_recall_at_k, precision_scores_at_k, f1_scores_at_k
