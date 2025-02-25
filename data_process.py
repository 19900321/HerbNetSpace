import pandas as pd
import pickle
import pandas as pd 
from collections import Counter
import random
import numpy as np
import collections

# graphy 
# graphy_pd = pd.read_csv('/home/yin/DREAMwalk-main/DREAMwalk-main/demo/DREAMwalk_file_2.csv')
# graphy_pd.astype({'node_1':str, 'node_2':str,'type':int, 'weight':float, 'id':int}).dtypes
# graphy_pd = graphy_pd.dropna()
# nodes_graphy = list(set(graphy_pd['node_1'].to_list() + graphy_pd['node_2'].to_list() ))

# # a.to_csv('/home/yin/DREAMwalk-main/DREAMwalk-main/demo/DREAMwalk_aimed_file.txt', sep='\t', index = None, header=None)

# # a = pd.read_csv('/home/yin/DREAMwalk-main/DREAMwalk-main/demo/hierarchy.csv')
# # a.to_csv('/home/yin/DREAMwalk-main/DREAMwalk-main/demo/hierarchy.txt', sep='\t', index=None)

# # b = pd.read_csv('/home/yin/DREAMwalk-main/DREAMwalk-main/demo/mtrees2024.txt', sep=';')
# # b.to_csv('/home/yin/DREAMwalk-main/DREAMwalk-main/demo/mtrees2024_prepared.txt', sep='\t', index=None)

# # keep only disease, drug, herb in both graphy and similarity
# # add herb in and regenerate herb
# similarity_pd  = pd.read_csv('/home/yin/DREAMwalk-main/DREAMwalk-main/demo/total_similarity_aimed.csv')
# nodes_simi = list(set(similarity_pd['source'].to_list() + similarity_pd['target'].to_list()))

# # keep only nodes in graph
# nodes_sim_in_grahy = [i for i in nodes_simi if i in nodes_graphy]
# similarity_pd_simple = similarity_pd.loc[(similarity_pd['source'].isin(nodes_sim_in_grahy )) & (similarity_pd['target'].isin(nodes_sim_in_grahy )),:]
# similarity_pd_simple.to_csv('/home/yin/DREAMwalk-main/DREAMwalk-main/demo/demo_similarty_graph_3.txt', sep='\t', index = None,header=None)

# # node type file
# node_type_pd = pd.read_csv('/home/yin/DREAMwalk-main/DREAMwalk-main/demo/node_type.csv')
# node_type_pd = node_type_pd.loc[node_type_pd['node'].isin(nodes_sim_in_grahy),:]
# nodes_in_types = list(set(node_type_pd['node'].to_list()))
# node_type_pd.to_csv('/home/yin/DREAMwalk-main/DREAMwalk-main/demo/node_type.txt', sep='\t', index = None)

## 1. prepared the input format files
def prepare_input_file():
    a = pd.read_csv('/home/yin/DREAMwalk-main/DREAMwalk-main/demo/DREAMwalk_file_2.csv')
    a.astype({'node_1':str, 'node_2':str,'type':int, 'weight':float, 'id':int}).dtypes
    a = a.dropna()
    a.to_csv('/home/yin/DREAMwalk-main/DREAMwalk-main/demo/DREAMwalk_aimed_file.txt', sep='\t', index = None, header=None)

    # add herb in and regenerate herb
    similarity_pd  = pd.read_csv('/home/yin/DREAMwalk-main/DREAMwalk-main/demo/total_similarity_aimed.csv')

    similarity_pd.to_csv('/home/yin/DREAMwalk-main/DREAMwalk-main/demo/demo_similarty_graph_3.txt', sep='\t', index = None,header=None)

    # delete disease/drug not in graph
    node_type_pd = pd.read_csv('/home/yin/DREAMwalk-main/DREAMwalk-main/demo/node_type.csv')
    node_type_pd = node_type_pd.dropna()
    node_type_pd.to_csv('/home/yin/DREAMwalk-main/DREAMwalk-main/demo/node_type.txt', sep='\t', index = None)

# save embedding
def save_key_files(transmatrixf, embeddingf, walk_pathf, save_embedding_f,save_simple_embedding_f,nodetypef ):
    ## save the transmatrix, walks, embeddings of nodes
    trans_matrix = pickle.load(open(transmatrixf,'rb'))
    embeddingf_matrix = pickle.load(open(embeddingf,'rb'))
    walk_path = pickle.load(open(walk_pathf,'rb'))
    # save as text
    mebedding_dict = pd.DataFrame.from_dict(embeddingf_matrix, orient='index')
    mebedding_dict['node'] = mebedding_dict.index

    # SAVE OUT
    node_type_pd = pd.read_csv(nodetypef, sep = '\t')
    merged_emebeding = pd.merge(node_type_pd, mebedding_dict, left_on= 'node', right_on = 'node', how = 'outer')
    merged_emebeding.to_csv(save_embedding_f, sep = '\t')
    simple_merged_emebeding = merged_emebeding.loc[merged_emebeding['type'].isin(['disease', 'drug', 'herb']), :]
    simple_merged_emebeding.to_csv(save_simple_embedding_f, sep = '\t')


# generate negative sampleS

def prepare_pairwise_combination(x,y, pairs_pos_drug_disease,i , recall_validation:True):
    
    pairs_neg_drug_disease = [[d, dis] for d in x for dis in y]
            
    # add negative pairs in drug and herb
    pairs_neg_drug_disease = [i for i in pairs_neg_drug_disease if i not in pairs_pos_drug_disease]
    neg_drug_disease_pd = pd.DataFrame(pairs_neg_drug_disease,
                                    columns=['drug', 'disease'])
    neg_drug_disease_pd['label'] = 0

    if not recall_validation:
        neg_drug_disease_pd_new_list = [neg_drug_disease_pd.sample(len(pairs_pos_drug_disease)) for i in range(i)]
    else:
        neg_drug_disease_pd_new_list = [neg_drug_disease_pd]
    return neg_drug_disease_pd_new_list

        
def prepare_negative_samples(drug_disease_f,embeddingf, disease_drug_label_f,i, recall_validation:True):
        # {'herb': 1140, 'drug': 1320, 'disease': 747})
    embeddingf_matrix = pickle.load(open(embeddingf,'rb'))
    drug_disease_pd = pd.read_csv(drug_disease_f, sep=' ')
    print(drug_disease_pd.columns)
    # drug_disease_pd.columns=['drug','disease','label']
    drug_disease_pd = drug_disease_pd.loc[(drug_disease_pd['drug'].isin(embeddingf_matrix))&(drug_disease_pd["disease"].isin(embeddingf_matrix)), :]
    drug_disease_pd = drug_disease_pd.drop_duplicates()
    
    # make all existing label as 1
    drug_disease_pd['label'] = 1
    pairs_pos_drug_disease = list(zip(drug_disease_pd['drug'],drug_disease_pd['disease']))

    # selelct herb, disease 
    diseases = drug_disease_pd['disease']
    drugs = drug_disease_pd['drug']
    neg_drug_disease_pd_list = prepare_pairwise_combination(drugs,diseases, pairs_pos_drug_disease, i,recall_validation)

    # merge positive and negative
    
    drug_disease_pd_total_list = []
    for i,neg_drug_disease_pd in enumerate(neg_drug_disease_pd_list):
        drug_disease_pd_total = pd.concat([drug_disease_pd, neg_drug_disease_pd])
        disease_drug_label_f_i = disease_drug_label_f + str(i) + '.txt'
        drug_disease_pd_total.to_csv(disease_drug_label_f_i, sep='\t', index = None)
        drug_disease_pd_total_list.append(drug_disease_pd_total)

    return drug_disease_pd_total_list



# save herb _label for combination
# drug_disease_pd_herb = drug_disease_pd_herb[['drug', 'disease', 'label_detail']].drop_duplicates()
# drug_disease_pd_herb.to_csv('/home/yin/DREAMwalk-main/DREAMwalk-main/demo/disease_label_herb.csv', index=None)
# a = Counter(drug_disease_pd_herb['label_detail'])
# drug_disease_pd_herb = drug_disease_pd_herb[['drug',  'label_detail']].drop_duplicates()
# a = Counter(drug_disease_pd_herb['label_detail'])
# load node type file and prepare as dictionary                             
    # node_type_pd = pd.read_csv(nodetypef, sep = '\t')
    # node_type_dict = dict(zip(node_type_pd['node'], node_type_pd['type']))
    # type_dict = dict(node_type_pd.groupby('type')['node'].apply(list))

    # # give the pairs type by drug /herb
    # drug_disease_pd['node_type'] = drug_disease_pd['drug'].apply(lambda x:node_type_dict.get(x))

    # # get the number pf pairs
    # print(Counter(drug_disease_pd['node_type']))
    # '''{'herb-disease': 8278, 'drug-disease': 2321})'''


# prepare all unknown herb_disease pairs
def pre_unknown_herb_disease(nodetypef, drug_disease_f, unknown_herb_disease_f):
    # read node in node type
    node_type_pd = pd.read_csv(nodetypef, sep = '\t')
    node_type_dict = dict(zip(node_type_pd['node'], node_type_pd['type']))
    type_dict = dict(node_type_pd.groupby('type')['node'].apply(list))

    # read nodes not in drug_herb_disease
    drug_disease_pd = pd.read_csv(drug_disease_f, sep='\t')

    # give the pairs type by drug /herb

    unkown_herbs = [h for h in type_dict['herb'] if h not in list(drug_disease_pd['drug'])]
    unkown_drug_disease_pairs = [(h, d) for h in unkown_herbs for d in type_dict['disease']]
    unkown_drug_disease_pairs_pd = pd.DataFrame(unkown_drug_disease_pairs, columns=['disease', 'drug'], index =None)
    unkown_drug_disease_pairs_pd.to_csv(unknown_herb_disease_f, index = None, sep= '\t')