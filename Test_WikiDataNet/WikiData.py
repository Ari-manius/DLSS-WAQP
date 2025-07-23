import pandas as pd
import json 
from statistics import mean
from sklearn.preprocessing import LabelEncoder
import umap
from sklearn.preprocessing import OneHotEncoder
import ast  
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

def parse_list(x):
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []  # fallback if parsing fails

def mean_list(class_list):
    mapped = [wiki_quality_cat_dict.get(c) for c in class_list if c in wiki_quality_cat_dict]
    return mean(mapped) if mapped else None

######
df_articles = pd.read_csv('data/wikipedia_articles_cleaned.csv')

article_TitleID_dict = dict(zip(df_articles["article"], df_articles["pageid"]))

with open("data/article_TitleID_dict.json", "w") as f:
    json.dump(article_TitleID_dict, f, indent=4)


#####
wiki_quality = ["FA", "FL", "FM", "A", "GA", "B", "C", "Start", "Stub", "List"]

wiki_quality_cat_dict = {
    "FA":9, 
    "FL":8,
    "FM":7,
    "A":6, 
    "GA":5,
    "B":4,
    "C":3, 
    "Start":2,
    "Stub":1,
    "List":0,     
}

wiki_quality_aggcat_dict = {
    "FA":2, 
    "FL":2,
    "FM":2,
    "A":1, 
    "GA":1,
    "B":1,
    "C":1, 
    "Start":0,
    "Stub":0,
    "List":0, 
}

df_articles['all_quality_classes'] = df_articles['all_quality_classes'].apply(parse_list)

df_articles['QC_num'] = df_articles['all_quality_classes'].apply(mean_list)
df_articles['QC_aggcat'] = df_articles['quality_class'].map(wiki_quality_aggcat_dict)
df_articles['QC_cat'] = df_articles['quality_class'].map(wiki_quality_cat_dict)
df_articles['QC_num_log'] = np.log1p(df_articles['QC_num'])

######
mapping = {
    'unprotected': 0,
    'semi_protected': 1,
    'protected': 2,
    'fully_protected': 3,
}
le = LabelEncoder()
df_articles["protection_status_encoded"] = le.fit_transform(df_articles["protection_status"])

######
df_articles['has_infobox_encoded'] = df_articles['has_infobox'].astype(int)

######
ohe = OneHotEncoder(handle_unknown='ignore')
X_ohe = ohe.fit_transform(df_articles[['assessment_source']])
reducer = umap.UMAP(n_components=2)
X_umap = reducer.fit_transform(X_ohe)
df_articles['assessment_source_umap_1'] = X_umap[:, 0]
df_articles['assessment_source_umap_2'] = X_umap[:, 1]

######
df_articles['category_list'] = df_articles['categories'].apply(ast.literal_eval)
mlb = MultiLabelBinarizer(sparse_output=True)
X_multi_hot = mlb.fit_transform(df_articles['category_list'])
reducer = umap.UMAP(n_components=3)
X_umap = reducer.fit_transform(X_multi_hot)
df_articles['categories_umap_1'] = X_umap[:, 0]
df_articles['categories_umap_2'] = X_umap[:, 1]
df_articles['categories_umap_3'] = X_umap[:, 2]


df_articles['template_list'] = df_articles['templates'].apply(ast.literal_eval)
mlb = MultiLabelBinarizer(sparse_output=True)
X_multi_hot = mlb.fit_transform(df_articles['template_list'])
reducer = umap.UMAP(n_components=3)
X_umap = reducer.fit_transform(X_multi_hot)
df_articles['templates_umap_1'] = X_umap[:, 0]
df_articles['templates_umap_2'] = X_umap[:, 1]
df_articles['templates_umap_3'] = X_umap[:, 2]


##############
df_ArticleTargetFeatures = df_articles[["pageid", 
                                        'QC_cat', 
                                        'QC_aggcat', 
                                        'QC_num',
                                        'QC_num_log', 
                                        "num_categories", 
                                        "num_links", 
                                        "page_length", 
                                        "num_references",  
                                        "num_sections", 
                                        "num_templates", 
                                        "has_infobox_encoded",
                                        "protection_status_encoded",  
                                        "assessment_source_umap_1", 
                                        "assessment_source_umap_2",
                                        "categories_umap_1",
                                        "categories_umap_2",
                                        "categories_umap_3",
                                        'templates_umap_1',
                                        'templates_umap_2',
                                        'templates_umap_3']]


df_ArticleTargetFeatures.to_json("data/wikidata_ready4net.json", orient="records", indent=4)