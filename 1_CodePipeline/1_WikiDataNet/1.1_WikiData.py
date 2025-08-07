import polars as pl
from statistics import mean
from sklearn.preprocessing import LabelEncoder
import umap
from sklearn.preprocessing import OneHotEncoder
import ast  
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
print("Loading")
df_articles = pl.read_csv('data/cleaned_articles_final.csv')

# Create article index dataframe
df_article_index = pl.DataFrame({
    'title': df_articles['article'],
    'id': range(len(df_articles)),
    'pageid': df_articles['pageid']
})

# Save as parquet
df_article_index.write_parquet("data/article_TitleIndex_dict.parquet")

#####
print("Processing Target")
wiki_quality = ["FA", "FL", "FM", "A", "GA", "B", "C", "Start", "Stub", "List"]

wiki_quality_cat_dict = {
    "FA": int(9), 
    "FL": int(8),
    "FM": int(7),
    "A": int(6), 
    "GA": int(5),
    "B": int(4),
    "C": int(3), 
    "Start": int(2),
    "Stub": int(1),
    "List": int(0),     
}

wiki_quality_aggcat_dict = {
    "FA":int(2), 
    "FL":int(2),
    "FM":int(2),
    "A":int(1), 
    "GA":int(1),
    "B":int(1),
    "C":int(1), 
    "Start":int(0),
    "Stub":int(0),
    "List":int(0), 
}

df_articles = df_articles.with_columns([
    pl.col('quality_class').replace(wiki_quality_aggcat_dict).cast(pl.Int64).alias('Target_QC_aggcat'),
    pl.col('quality_class').replace(wiki_quality_cat_dict).cast(pl.Int64).alias('Target_QC_cat')
])

df_articles = df_articles.with_columns([
    pl.col('Target_QC_cat').log1p().alias('Target_QC_numlog')
])

######
print("Encoding")
mapping = {
    'unprotected': 0,
    'semi_protected': 1,
    'protected': 2,
    'fully_protected': 3,
}
le = LabelEncoder()
encoded_protection = le.fit_transform(df_articles["protection_status"].to_numpy())
df_articles = df_articles.with_columns([
    pl.Series("protection_status_encoded", encoded_protection)
])

######
df_articles = df_articles.with_columns([
    pl.col('has_infobox').cast(pl.Int64).alias('has_infobox_encoded')
])

###### 
# Optionally remove this part? 
print("One Hot and Dimensionality Reduction")
ohe = OneHotEncoder(handle_unknown='ignore')
X_ohe = ohe.fit_transform(df_articles[['assessment_source']].to_numpy())
reducer = umap.UMAP(n_components=3)
X_umap = reducer.fit_transform(X_ohe)
df_articles = df_articles.with_columns([
    pl.Series("assessment_source_umap_1", X_umap[:, 0]),
    pl.Series("assessment_source_umap_2", X_umap[:, 1]),
    pl.Series("assessment_source_umap_3", X_umap[:, 2]),
])

##############
print("Saving")
df_ArticleTargetFeatures = df_articles[["pageid", 
                                        'Target_QC_cat', 
                                        'Target_QC_aggcat', 
                                        'Target_QC_numlog', 
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
                                        "assessment_source_umap_3"]]

df_ArticleTargetFeatures.write_parquet("data/wikidata_ready4net.parquet")