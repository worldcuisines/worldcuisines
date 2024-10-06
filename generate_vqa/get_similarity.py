import pandas as pd
import numpy as np 
import json
import logging
import argparse

import dask.dataframe as dd

from utils import *

def map_similarity_to_ids(similarity_df, food_df):
    # Create a mapping of food names to food IDs
    name_to_id_map = pd.Series(food_df['food_id'].values, index=food_df['Name']).to_dict()

    # Filter out any names in similarity_df that are not present in food_df
    valid_names = similarity_df.index.intersection(name_to_id_map.keys())

    # Subset similarity_df to only include valid names
    new_similarity_df = similarity_df.loc[valid_names, valid_names]

    # Replace index and column names with their corresponding food IDs
    new_similarity_df.index = new_similarity_df.index.map(name_to_id_map)
    new_similarity_df.columns = new_similarity_df.columns.map(name_to_id_map)

    return new_similarity_df

def helper_find_category(food_df, food_id, top_similar, category_name):
    food_category = food_df.loc[food_df['food_id'] == food_id, category_name].values[0]
    similar_foods_with_same_fine_category = []
    try:
        food_category = json.loads(food_category) # Convert into list
        for similar_food_id in top_similar.index:
            similar_food_category = food_df.loc[food_df['food_id'] == similar_food_id, category_name].values[0]
            
            if food_category is None or similar_food_category is None:
                continue
            
            try:
                similar_food_category = json.loads(similar_food_category) # Convert into list
            except Exception:
                continue
            
            # Check if the categories match
            if set(food_category) == set(similar_food_category):
                similar_foods_with_same_fine_category.append(similar_food_id)
            
            # Stop once we find k similar foods with the same category
            if len(similar_foods_with_same_fine_category) == TOP_K:
                break
    except Exception:
        pass
    
    return similar_foods_with_same_fine_category

def find_top_k_similar_foods(food_id, similarity_df, food_df):
    # Sort the similarity row for the current food_id
    food_similarity = similarity_df.loc[food_id].copy()
    top_similar = food_similarity.sort_values(ascending=False)
    top_similar = top_similar[top_similar.index != food_id]
    
    # Get the top k most similar foods (without category check)
    top_k_similar = top_similar.head(TOP_K).index.tolist()

    # Filter by fine/coarse-grained category
    similar_foods_with_same_fine_category = helper_find_category(food_df, food_id, top_similar, 'Fine-grained categories')
    similar_foods_with_same_coarse_category = helper_find_category(food_df, food_id, top_similar, 'Coarse-grained categories')

    return top_k_similar, similar_foods_with_same_fine_category, similar_foods_with_same_coarse_category

def get_top_k_similar_foods(food_df_path, similarity_pq_path):
    similarity_df = pd.read_parquet(similarity_pq_path)
    food_df = pd.read_csv(food_df_path)
    food_df['food_id'] = np.arange(len(food_df)) # In case it does not have any ID

    # Apply the mapping function
    similarity_df_with_ids = map_similarity_to_ids(similarity_df, food_df)

    # Convert Pandas DataFrame to Dask DataFrame for parallel execution
    dask_df = dd.from_pandas(food_df, npartitions=N_THREADS)

    # Define the schema (meta) for the result of find_top_k_similar_foods
    meta = pd.DataFrame({
        'Top_k_Similar_Foods': pd.Series(dtype='object'),
        'Top_k_Same_Fine_Category': pd.Series(dtype='object'),
        'Top_k_Same_Coarse_Category': pd.Series(dtype='object')
    })
    
    # Parallel function for finding top k similar foods
    def parallel_find_top_k_similar_foods(row):
        top_k_similar, top_k_fine, top_k_coarse = find_top_k_similar_foods(row['food_id'], similarity_df_with_ids, food_df)
        return pd.Series({
            'Top_k_Similar_Foods': top_k_similar,
            'Top_k_Same_Fine_Category': top_k_fine,
            'Top_k_Same_Coarse_Category': top_k_coarse
        })

    # Apply the function in parallel across partitions
    results = dask_df.apply(parallel_find_top_k_similar_foods, axis=1, meta=meta)
    results = results.compute()
    food_df[['Top_k_Similar_Foods', 'Top_k_Same_Fine_Category', 'Top_k_Same_Coarse_Category']] = results

    food_df.to_csv(food_df_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Arguments for experiment regressor mode
    parser.add_argument('-f', '--food_df_path', type=str, required=False, default=os.path.join(RESOURCE_DIR, "food_df.csv"), help="Path used to load food dataframe")
    parser.add_argument('-s', '--similarity_pq_path', type=str, required=False, default=os.path.join(RESOURCE_DIR, "similarity.parquet"), help="Path used to load the similarity matrix")
    args = parser.parse_args()
    
    os.makedirs(RESOURCE_DIR, exist_ok=True)
    
    get_top_k_similar_foods(args.food_df_path, args.similarity_pq_path)

    