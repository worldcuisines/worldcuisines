import os
import random
import ast
import pandas as pd
import numpy as np
import argparse
import re
import time
from math import ceil

from utils import (
    RANDOM_SEED,
    RESOURCE_DIR,
    TOP_K, MAX_ANS_ALL, MAX_ANS_FROM_FINEGRAINED, MAX_ANS_FROM_COARSEGRAINED,
    PROMPT_EVAL_PORTION, NUM_MAX_DISHES_EVAL, CONTEXT_TYPE_ACTION, ODDS_ADVERSARIAL,
    ALL_LANGUAGES, LOCATION_MAPPING, LANGUAGE_CODE_MAPPING,
)
import logging
logging.basicConfig(level=logging.DEBUG)


def sample_an_image(food_id, food_df):
    # Get the image list for the food_id from food_df
    image_list = food_df.loc[food_df['food_id'] == food_id, 'ImageUrls'].values[0]
    try:
        image_list = ast.literal_eval(image_list)      
        if isinstance(image_list, list) and len(image_list) > 0:
            # Randomly sample an index from the image list
            sampled_index = random.randint(0, len(image_list) - 1)
            sampled_image_url = image_list[sampled_index]
            return sampled_index, sampled_image_url
        else:
            return None, None
    except Exception:
        return None, None
    
def sample_an_answer(row, food_df):
    if row['action'] == 'Name':
        food_id = row['food_id']
        food_name = food_df.loc[food_df['food_id'] == food_id, 'Name'].values[0]
        return food_name
    else:
        food_id = row['food_id']
        countries = food_df.loc[food_df['food_id'] == food_id, 'Countries'].values[0]
        countries = ast.literal_eval(countries)  
        # Sample random countries
        country = countries[random.randint(0, len(countries) - 1)]
        return country

def get_food_indices_given_row(row):
    # Get indices for the nearest answers
    if type(row[f'Top_{TOP_K}_Same_Fine_Category']) == str:
        index_map = {
            f'Top_{TOP_K}_Same_Fine_Category': row[f'Top_{TOP_K}_Same_Fine_Category'],
            f'Top_{TOP_K}_Same_Coarse_Category': row[f'Top_{TOP_K}_Same_Coarse_Category'],
            f'Top_{TOP_K}_Similar_Foods': row[f'Top_{TOP_K}_Similar_Foods']
        }
    else:
        index_map = {
            f'Top_{TOP_K}_Same_Fine_Category': row[f'Top_{TOP_K}_Same_Fine_Category'].values[0],
            f'Top_{TOP_K}_Same_Coarse_Category': row[f'Top_{TOP_K}_Same_Coarse_Category'].values[0],
            f'Top_{TOP_K}_Similar_Foods': row[f'Top_{TOP_K}_Similar_Foods'].values[0]
        }
        
    try:
        index_map[f'Top_{TOP_K}_Same_Fine_Category'] = ast.literal_eval(index_map[f'Top_{TOP_K}_Same_Fine_Category'])
    except Exception:
        index_map[f'Top_{TOP_K}_Same_Fine_Category'] = []
        
    try:
        index_map[f'Top_{TOP_K}_Same_Coarse_Category'] = ast.literal_eval(index_map[f'Top_{TOP_K}_Same_Coarse_Category'])
    except Exception:
        index_map[f'Top_{TOP_K}_Same_Coarse_Category'] = []


    try:
        index_map[f'Top_{TOP_K}_Same_Similar_Foods'] = ast.literal_eval(index_map[f'Top_{TOP_K}_Similar_Foods'])
    except Exception:
        index_map[f'Top_{TOP_K}_Same_Similar_Foods'] = []
        
    return index_map

def shuffle_answers_with_correct_index(wrong_answers, correct_answer):
    shuffled_answers = wrong_answers.copy()
    correct_index = random.randint(0, len(shuffled_answers))
    shuffled_answers.insert(correct_index, correct_answer)
    return shuffled_answers, correct_index

def generate_translate_food_alias_aware(row, food_raw_df, all_indices_sets, languages_used):
    first_indices = all_indices_sets[0]
    second_indices = all_indices_sets[1]
    third_indices = all_indices_sets[2]
    
    adversarial_answers = []

    # Find indices that contain all aliases for each level
    for indices in all_indices_sets:
        for index in indices:
            # Assuming 'Alias' is a string representation of a dictionary
            alias_dict = ast.literal_eval(food_raw_df.loc[index, 'Alias'])
            alias_dict = {lang.split(' - ')[0]: alias_dict[lang] for lang in alias_dict}

            for lang in languages_used:
                translation = food_raw_df.loc[index, 'Name']
                no_alias = True
                alias_lang = lang.split("_")[0]
                if alias_lang in alias_dict.keys():
                    translation = alias_dict[alias_lang]
                    no_alias = False
                
                if no_alias:
                    break
            
            # Append, no need fancy randomization
            adversarial_answers.append(index)
            
            if len(adversarial_answers) == MAX_ANS_ALL:
                break   
    
        if len(adversarial_answers) == MAX_ANS_ALL:
            break   
    
    # Get 2 answers from food in fine-grained
    if len(adversarial_answers) != MAX_ANS_ALL and len(first_indices) > 0:
        sample_size = min(MAX_ANS_FROM_FINEGRAINED, len(first_indices))
        random_fine_answers = np.random.choice(list(first_indices), size=sample_size, replace=False)
        adversarial_answers.extend(random_fine_answers.tolist())
    
    # Get 1 answer from food in coarse-grained
    if len(adversarial_answers) != MAX_ANS_ALL and len(second_indices) > 0:
        sample_size = min(MAX_ANS_FROM_COARSEGRAINED - len(adversarial_answers), len(second_indices))
        random_coarse_answers = np.random.choice(list(second_indices), size=sample_size, replace=False)
        adversarial_answers.extend(random_coarse_answers.tolist())
    
    # Get remaining answer 
    if len(adversarial_answers) != MAX_ANS_ALL and len(third_indices) > 0:
        sample_size = min(MAX_ANS_ALL - len(adversarial_answers), len(third_indices))
        random_remain_answers = np.random.choice(list(third_indices), size=sample_size, replace=False)
        adversarial_answers.extend(random_remain_answers.tolist())

    # Convert id to name and return only the required number of answers
    adversarial_answers = [food_raw_df.loc[idx, 'Name'] for idx in adversarial_answers]
    shuffled_answers, answer_index = shuffle_answers_with_correct_index(adversarial_answers, row['answer'])
    
    # Convert each answer into each language name using aliasing
    translation_dict = {}
    lang_status = 'full'
    for ans in shuffled_answers:
        index = food_raw_df[food_raw_df['Name'].str.strip() == ans.strip()].index[0]
        # Assuming 'Alias' is a string representation of a dictionary
        alias_dict = ast.literal_eval(food_raw_df.loc[index, 'Alias'])
        alias_dict = {lang.split(' - ')[0]: alias_dict[lang] for lang in alias_dict}

        for lang in languages_used:
            translation = food_raw_df.loc[index, 'Name']
            no_alias = True
            alias_lang = lang.split("_")[0]
            if alias_lang in alias_dict.keys():
                translation = alias_dict[alias_lang]
                no_alias = False
                    
            if no_alias:
                lang_status = 'mixed'
    
            # If the language is not already in the dictionary, initialize it as an empty list
            if lang not in translation_dict:
                translation_dict[lang] = []
            
            # Append the translation for the current answer
            translation_dict[lang].append(translation)
    
    return translation_dict, answer_index, lang_status

def generate_translate_food_default(row, food_raw_df, all_indices_sets, languages_used):
    first_indices = all_indices_sets[0]
    second_indices = all_indices_sets[1]
    third_indices = all_indices_sets[2]
    
    adversarial_answers = []

    # Get 2 answers from food in fine-grained
    if len(first_indices) > 0:
        sample_size = min(MAX_ANS_FROM_FINEGRAINED, len(first_indices))
        random_fine_answers = np.random.choice(list(first_indices), size=sample_size, replace=False)
        adversarial_answers.extend(random_fine_answers.tolist())
    
    # Get 1 answer from food in coarse-grained
    if len(second_indices) > 0:
        sample_size =  min(MAX_ANS_FROM_COARSEGRAINED - len(adversarial_answers), len(second_indices))
        random_coarse_answers = np.random.choice(list(second_indices), size=sample_size, replace=False)
        adversarial_answers.extend(random_coarse_answers.tolist())
    
    # Get remaining answer 
    if len(third_indices) > 0:
        sample_size = min(MAX_ANS_ALL - len(adversarial_answers), len(third_indices))
        random_remain_answers = np.random.choice(list(third_indices), size=sample_size, replace=False)
        adversarial_answers.extend(random_remain_answers.tolist())

    # Convert id to name and return only the required number of answers
    adversarial_answers = [food_raw_df.loc[idx, 'Name'] for idx in adversarial_answers]
    shuffled_answers, answer_index = shuffle_answers_with_correct_index(adversarial_answers, row['answer'])
    
    # Convert each answer into each language name using aliasing
    translation_dict = {}
    lang_status = 'full'
    for ans in shuffled_answers:
        index = food_raw_df[food_raw_df['Name'].str.strip() == ans.strip()].index[0]
        # Assuming 'Alias' is a string representation of a dictionary
        alias_dict = ast.literal_eval(food_raw_df.loc[index, 'Alias'])
        alias_dict = {lang.split(' - ')[0]: alias_dict[lang] for lang in alias_dict}

        for lang in languages_used:
            translation = food_raw_df.loc[index, 'Name']
            no_alias = True
            alias_lang = lang.split("_")[0]
            if alias_lang in alias_dict.keys():
                translation = alias_dict[alias_lang]
                no_alias = False
                    
            if no_alias:
                lang_status = 'mixed'
    
            # If the language is not already in the dictionary, initialize it as an empty list
            if lang not in translation_dict:
                translation_dict[lang] = []
            
            # Append the translation for the current answer
            translation_dict[lang].append(translation)
    
    return translation_dict, answer_index, lang_status

def generate_location_base_answers(row, food_raw_df, all_indices_sets):
    first_indices = all_indices_sets[0]
    second_indices = all_indices_sets[1]
    third_indices = all_indices_sets[2]
    
    # Get the Location/Area, ensure that later no answer should be either of these
    possible_base_answers = []
    try:
        cur_answers = row['Area']
        cur_answers = ast.literal_eval(cur_answers)
        possible_base_answers.extend(cur_answers)
    except Exception:
        pass
    
    try:
        cur_answers = row['Countries']
        cur_answers = ast.literal_eval(cur_answers)
        possible_base_answers.extend(cur_answers)
    except Exception:
        pass
    
    adversarial_answers = []
    
    # Get 2 answers from (random) countries in fine-grained
    fine_grained_answers = set()
    fine_foods = food_raw_df[food_raw_df['food_id'].isin(first_indices)].reset_index(drop=True)
    for _, fine_row in fine_foods.iterrows():
        fine_countries = fine_row['Countries']
        try:
            fine_countries = ast.literal_eval(fine_countries)
            fine_grained_answers.update(fine_countries)
        except Exception:
            continue
        
    # Remove elements in fine_grained_answers that are also in possible_base_answers
    fine_grained_answers = fine_grained_answers - set(possible_base_answers)
    if len(fine_grained_answers) > 0:
        # Randomly sample up to 2 answers from fine_grained_answers
        sample_size = min(MAX_ANS_FROM_FINEGRAINED, len(fine_grained_answers))
        random_fine_answers = np.random.choice(list(fine_grained_answers), size=sample_size, replace=False)
        adversarial_answers.extend(random_fine_answers.tolist())
        
    # Get max 3 answers in adversarial answers (random) countries in coarse-grained
    coarse_grained_answers = set()
    coarse_foods = food_raw_df[food_raw_df['food_id'].isin(second_indices)].reset_index(drop=True)
    for _, coarse_row in coarse_foods.iterrows():
        coarse_countries = coarse_row['Countries']
        try:
            coarse_countries = ast.literal_eval(coarse_countries)
            coarse_grained_answers.update(coarse_countries)
        except Exception:
            continue
        
    # Remove elements in fine_grained_answers that are also in possible_base_answers and current adversarial answers
    coarse_grained_answers = coarse_grained_answers - set(possible_base_answers) - set(adversarial_answers)
    if len(coarse_grained_answers) > 0:
        # Randomly sample up to 3 answers from coarse_grained_answers
        sample_size = min(MAX_ANS_FROM_COARSEGRAINED - len(adversarial_answers), len(coarse_grained_answers))
        random_coarse_answers = np.random.choice(list(coarse_grained_answers), size=sample_size, replace=False)
        adversarial_answers.extend(random_coarse_answers.tolist())
        
    # Get remaining answers in adversarial answers (random) countries
    remain_answers = set()
    remain_foods = food_raw_df[food_raw_df['food_id'].isin(third_indices)].reset_index(drop=True)
    for _, remain_row in remain_foods.iterrows():
        remain_countries = remain_row['Countries']
        try:
            remain_countries = ast.literal_eval(remain_countries)
            remain_answers.update(remain_countries)
        except Exception:
            continue
        
    # Remove elements in fine_grained_answers that are also in possible_base_answers and current adversarial answers
    remain_answers = remain_answers - set(possible_base_answers) - set(adversarial_answers)
    if len(remain_answers) > 0:
        # Randomly sample up to 4 answers from coarse_grained_answers
        random_remain_answers = np.random.choice(list(remain_answers), min(MAX_ANS_ALL - len(adversarial_answers),
                                                 len(remain_answers)))
        adversarial_answers.extend(random_remain_answers.tolist())  # Convert NumPy array to list before extending
    
    # Not yet full? sample from the rest
    if len(adversarial_answers) != MAX_ANS_ALL:
        answers_unite = fine_grained_answers.union(coarse_grained_answers).union(remain_answers) - set(adversarial_answers) - set(possible_base_answers)
        sample_size = min(MAX_ANS_ALL - len(adversarial_answers), len(answers_unite))
        random_unite_answers = np.random.choice(list(answers_unite), size=sample_size, replace=False)
        adversarial_answers.extend(random_unite_answers.tolist())
    
    # Edge case, all adversarial answers cannot be found for some reason
    if len(adversarial_answers) != MAX_ANS_ALL:
        logging.warning(f"For some reason number of adversarial answers for location is not {MAX_ANS_ALL}; Will resample!")
        return [row['answer']], 0

    # Return only the required number of answers
    shuffled_answers, answer_index = shuffle_answers_with_correct_index(adversarial_answers, row['answer'])
    
    return shuffled_answers, answer_index

def translate_location_languages(all_answers, location_cuisine_df, languages_used):
    translation_dict = {}
    for ans in all_answers:
        try:
            index = location_cuisine_df[location_cuisine_df['base_key'].str.strip() == ans.strip()].index[0]
        except Exception:
            logging.info(f"Error at {ans}")
            raise RuntimeError

        # Create a dictionary for the translations
        translations = {
            lang: location_cuisine_df.loc[index, LOCATION_MAPPING[LANGUAGE_CODE_MAPPING[lang]]] 
            for lang in languages_used
        }
        
        # Update translation_dict for each answer
        for lang in translations:
            # If the language is not already in the dictionary, initialize it as an empty list
            if lang not in translation_dict:
                translation_dict[lang] = []
            
            # Append the translation for the current answer
            translation_dict[lang].append(translations[lang])

    return translation_dict
    
def generate_answers_per_language(row, food_raw_df, location_cuisine_df,
                                  all_indices_sets, languages_used, alias_aware=False):
    if row['action'] == 'Name':
        # If the question is about food name, cautious about aliases
        if alias_aware:
            translation_dict, correct_index, lang_status = generate_translate_food_alias_aware(row, food_raw_df, all_indices_sets, languages_used)
        else:
            translation_dict, correct_index, lang_status = generate_translate_food_default(row, food_raw_df, all_indices_sets, languages_used)
        return translation_dict, correct_index, lang_status
    elif row['action'] == 'Location':
        # Otherwise, pretty sure other languages will have the translation
        all_answers, correct_index = generate_location_base_answers(row, food_raw_df, all_indices_sets)
        if len(all_answers) != (MAX_ANS_ALL + 1):
            return {}, -1, "failed"
        translation_dict = translate_location_languages(all_answers, location_cuisine_df, languages_used)
        return translation_dict, correct_index, "full" # Assume always successful translation
    else:
        raise NotImplementedError(f"Answer type {row['action']} has not yet been implemented")

def get_nearest_answers(filtered_sampled_df, food_raw_df, food_cleaned_df, location_cuisine_df, languages_used, alias_aware=False):
    # Iterate through each row in the subset_df
    for idx, row in filtered_sampled_df.iterrows():
        food_id = row['food_id']
        food_row = food_cleaned_df.loc[food_cleaned_df['food_id'] == food_id]
        index_map = get_food_indices_given_row(food_row)

        all_indices_sets = []

        # Create the first set from the first list
        first_indices = set(index_map.get(f'Top_{TOP_K}_Same_Fine_Category', []))
        all_indices_sets.append(first_indices)

        # Create the second set from the second list while removing duplicates from the first
        second_indices = set(index_map.get(f'Top_{TOP_K}_Same_Coarse_Category', [])) - first_indices
        all_indices_sets.append(second_indices)

        # Create the third set from the third list while removing duplicates from the first and second
        third_indices = set(index_map.get(f'Top_{TOP_K}_Same_Similar_Foods', [])) - first_indices - second_indices
        all_indices_sets.append(third_indices)
        
        # Get answers per languages
        answers_dict, correct_index, lang_status = generate_answers_per_language(row, food_raw_df, location_cuisine_df,
                                                              all_indices_sets, languages_used, alias_aware)
        for lang, answers in answers_dict.items():
            filtered_sampled_df.at[idx, f'answer_{lang}'] = str(answers)
        filtered_sampled_df.at[idx, 'correct_index'] = int(correct_index)
        filtered_sampled_df.at[idx, 'lang_status'] = lang_status
        
    further_filt_sampled_df = filtered_sampled_df[filtered_sampled_df['lang_status'] != 'failed']
    
    return further_filt_sampled_df

def generate_duplicated_prompt_indices(query_context_df):
    # Function to find indices of duplicates in each column
    duplicated_dict = {}
    for column in query_context_df.columns:
        # Get the indices of duplicate values
        duplicate_indices = query_context_df[query_context_df[column].duplicated(keep=False)].index.tolist()
        if duplicate_indices:
            duplicated_dict[column] = duplicate_indices  # Store indices of duplicates

    duplicated_dict.pop('setup')
    duplicated_dict.pop('template')
    duplicated_dict.pop('prompt_type')
    
    # Union the indices from all columns
    all_duplicated_indices = set()
    for indices in duplicated_dict.values():
        all_duplicated_indices.update(indices)
        
    reserved_train_qc_df = query_context_df.loc[list(all_duplicated_indices)]

    return reserved_train_qc_df

def eval_test_possible_combinations(cleaned_food_df, query_context_df, is_eval):
    """
    1) Small eval set (WCVQA 12k):
    task1 (name): 100 dish x 3 prompt (1 w/o 1 w/ context 1 adversarial) x 30 languages = 9k 
    task2 (location): 100 dish x 1 prompt x 30 languages = 3k

    2) Large eval set (WCVQA 60k)
    task1: 500 dish x 3 prompt x 30 languages = 45k
    task2: 500 dish x 1 prompt x 30 languages = 15k

    Training: (WCVQA 1M)
    task1: 1800 dish x 10 prompt (5 w/o 5 w/ context 5 adversarial) x 30 languages = 810k
    task2: 1800 dish x 5 prompt x 30 languages = 270k
    """
    # Sampled dish max
    sampled_food_df = cleaned_food_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    if is_eval:
        sampled_food_df = sampled_food_df[:NUM_MAX_DISHES_EVAL]
    else:
        sampled_food_df = sampled_food_df[NUM_MAX_DISHES_EVAL:]
    
    # Sampled prompt max per type
    concat_type_dfs = []
    
    reserved_train_qc_df = generate_duplicated_prompt_indices(query_context_df)
    
    for type in range(1, 5):
        qc_type_df = query_context_df[query_context_df['prompt_type'] == type]
        reserved_type_df = reserved_train_qc_df[reserved_train_qc_df['prompt_type'] == type]
        available_type_df = qc_type_df[~qc_type_df.index.isin(reserved_type_df.index)].sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        need_train_left = int((1 - PROMPT_EVAL_PORTION) * len(qc_type_df)) - len(reserved_type_df)
        
        if is_eval:
            if need_train_left > 0:
                sampled_qc_df = available_type_df[need_train_left:]
            else:
                sampled_qc_df = available_type_df
                assert(len(sampled_qc_df) != 0) # In case all are taken by training
                
            concat_type_dfs.append(sampled_qc_df)
        else:
            concat_type_dfs.append(reserved_type_df)
            if need_train_left > 0:
                sampled_qc_df = available_type_df[:need_train_left]
                concat_type_dfs.append(sampled_qc_df)
 
    sampled_qc_df = pd.concat(concat_type_dfs)
    
    return sampled_food_df, sampled_qc_df

def traditionally_sampled(cleaned_food_df, query_context_df,
                          n_dish_max, n_prompt_max_type1, n_prompt_max_type2, n_prompt_max_type3, n_prompt_max_type4):
    # Sampled dish max
    n_dish_max = len(cleaned_food_df) if n_dish_max <= 0 else min(n_dish_max, len(cleaned_food_df))
    sampled_food_df = cleaned_food_df.sample(n=n_dish_max, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Sampled prompt max per type
    concat_type_dfs = []
    if n_prompt_max_type1 != 0:
        sampled_qc_type1_df = query_context_df[query_context_df['prompt_type'] == 1]
        n_prompt_max_type1 = len(sampled_qc_type1_df) if n_prompt_max_type1 <= 0 else min(n_prompt_max_type1, len(sampled_qc_type1_df))
        sampled_qc_type1_df = sampled_qc_type1_df.sample(n=n_prompt_max_type1, random_state=RANDOM_SEED).reset_index(drop=True)
        concat_type_dfs.append(sampled_qc_type1_df)

    if n_prompt_max_type2 != 0:
        sampled_qc_type2_df = query_context_df[query_context_df['prompt_type'] == 2]
        n_prompt_max_type2 = len(sampled_qc_type2_df) if n_prompt_max_type2 <= 0 else min(n_prompt_max_type2, len(sampled_qc_type2_df))
        sampled_qc_type2_df = sampled_qc_type2_df.sample(n=n_prompt_max_type2, random_state=RANDOM_SEED).reset_index(drop=True)
        concat_type_dfs.append(sampled_qc_type2_df)

    if n_prompt_max_type3 != 0:
        sampled_qc_type3_df = query_context_df[query_context_df['prompt_type'] == 3]
        n_prompt_max_type3 = len(sampled_qc_type3_df) if n_prompt_max_type3 <= 0 else min(n_prompt_max_type3, len(sampled_qc_type3_df))
        sampled_qc_type3_df = sampled_qc_type3_df.sample(n=n_prompt_max_type3, random_state=RANDOM_SEED).reset_index(drop=True)
        concat_type_dfs.append(sampled_qc_type3_df)

    if n_prompt_max_type4 != 0:
        sampled_qc_type4_df = query_context_df[query_context_df['prompt_type'] == 4]
        n_prompt_max_type4 = len(sampled_qc_type4_df) if n_prompt_max_type4 <= 0 else min(n_prompt_max_type4, len(sampled_qc_type4_df))
        sampled_qc_type4_df = sampled_qc_type4_df.sample(n=n_prompt_max_type4, random_state=RANDOM_SEED).reset_index(drop=True)
        concat_type_dfs.append(sampled_qc_type4_df)
 
    sampled_qc_df = pd.concat(concat_type_dfs)
    
    return sampled_food_df, sampled_qc_df
    
def filtered_combinations(sampled_food_df, sampled_qc_df, cleaned_food_df,
                          location_cuisine_df, n_dish_max):
    # Merge sampled_food_df and query_context_df using cross join
    food_query_combinations = pd.merge(
        sampled_food_df[['food_id', 'ImageUrls']],  # Now includes imageindex
        sampled_qc_df[['prompt_id', 'prompt_type']],
        how='cross'
    )
    
    # Drop rows in full_df where it contains invalid prompt
    sampled_qc_df['action'] = sampled_qc_df['template'].map(CONTEXT_TYPE_ACTION)
    action_df = sampled_qc_df[['prompt_id', 'action']]
    
    init_df = pd.merge(food_query_combinations, action_df, on='prompt_id', how='left') # Get action_df for later
    
    # Filter dataset where location is not proper (not in location_cuisine_df)
    standardized_locations = location_cuisine_df['base_key'].to_list()
    standardized_locations = [location.strip() for location in standardized_locations]
    if 'global' in standardized_locations:
        standardized_locations.remove('global')
    if 'Global' in standardized_locations:
        standardized_locations.remove('Global') 
    
    # First, join filtered_full_df with food_df to bring in the Countries column
    # Assuming food_id is the key for joining
    merged_df = pd.merge(init_df, cleaned_food_df[['food_id', 'Countries']], on='food_id', how='left')
    merged_df['Countries'] = merged_df['Countries'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Function to filter and reduce countries
    def filter_and_reduce_countries(row):
        countries = row['Countries']
        # Filter countries to only keep those that are in standardized_locations
        reduced_countries = [country.strip() for country in countries if country.strip() in standardized_locations]
        # Check if reduced countries is not empty
        return reduced_countries if reduced_countries else None  # Or return an empty list if you prefer

    # Apply the filtering and reducing function
    merged_df['Countries'] = merged_df.apply(filter_and_reduce_countries, axis=1)

    # Remove rows where 'Countries' is None or an empty list
    filtered_init_df = merged_df[merged_df['Countries'].apply(lambda x: x is not None and len(x) > 0)]
    filtered_init_df = filtered_init_df.drop(columns=['Countries'])

    # If Answer is Location, then sample different locations
    sampled_answers = []
    for _, row in filtered_init_df.iterrows():
        # Sample answer if it questions about Location
        sampled_answer = sample_an_answer(row, sampled_food_df)
        sampled_answers.append(sampled_answer)
    filtered_init_df['answer'] = sampled_answers
    
    # Explode sampled_food_df by creating an index for the ImageUrls column
    filtered_init_df['ImageUrls'] = filtered_init_df['ImageUrls'].apply(lambda x: ast.literal_eval(x))
    filtered_init_df = filtered_init_df[filtered_init_df['ImageUrls'].apply(len) > 0]
    unique_food_ids = filtered_init_df['food_id'].unique()[:n_dish_max]
    filtered_init_df = filtered_init_df[filtered_init_df['food_id'].isin(unique_food_ids)]
    filtered_init_df['image_index'] = filtered_init_df['ImageUrls'].apply(lambda x: list(range(len(x))))
    final_df = filtered_init_df.explode(['ImageUrls', 'image_index'])
    final_df = final_df.rename(columns={"ImageUrls": "image_url"})
    
    # Join has been performed, we can then add back the dropped information
    final_df = pd.merge(final_df, food_cleaned_df[['food_id', 'Coarse-grained categories','Fine-grained categories','Cuisines','Associated Cuisines','Area','Countries','Regions','Text Description']], how='left', on='food_id')

    return final_df

def generate_adversarial_loc_cuis_for_prompt(row, rnd_int, adver_or_not, location_cuisine_df):
    index_map = get_food_indices_given_row(row)

    all_indices_sets = []

    # Create the first set from the first list
    first_indices = set(index_map.get(f'Top_{TOP_K}_Same_Fine_Category', []))
    all_indices_sets.append(first_indices)

    # Create the second set from the second list while removing duplicates from the first
    second_indices = set(index_map.get(f'Top_{TOP_K}_Same_Coarse_Category', [])) - first_indices
    all_indices_sets.append(second_indices)

    # Create the third set from the third list while removing duplicates from the first and second
    third_indices = set(index_map.get(f'Top_{TOP_K}_Same_Similar_Foods', [])) - first_indices - second_indices
    all_indices_sets.append(third_indices)
    
    first_indices = all_indices_sets[0]
    
    # Get the Location/Area, ensure that later no answer should be either of these
    possible_base_answers = []
    try:
        cur_answers = row['Area']
        cur_answers = ast.literal_eval(cur_answers)
        possible_base_answers.extend(cur_answers)
    except Exception:
        pass
    
    try:
        cur_answers = row['Countries']
        cur_answers = ast.literal_eval(cur_answers)
        possible_base_answers.extend(cur_answers)
    except Exception:
        pass
    
    # Get all answers from (random) countries in fine-grained
    fine_grained_answers = set()
    fine_foods = food_raw_df[food_raw_df['food_id'].isin(first_indices)].reset_index(drop=True)
    for _, fine_row in fine_foods.iterrows():
        fine_countries = fine_row['Countries']
        try:
            fine_countries = ast.literal_eval(fine_countries)
            fine_grained_answers.update(fine_countries)
        except Exception:
            continue
        
    # Remove elements in fine_grained_answers that are also in possible_base_answers
    fine_grained_answers = fine_grained_answers - set(possible_base_answers)
    
    # Get max 3 answers in adversarial answers (random) countries in coarse-grained
    coarse_grained_answers = set()
    coarse_foods = food_raw_df[food_raw_df['food_id'].isin(second_indices)].reset_index(drop=True)
    for _, coarse_row in coarse_foods.iterrows():
        coarse_countries = coarse_row['Countries']
        try:
            coarse_countries = ast.literal_eval(coarse_countries)
            coarse_grained_answers.update(coarse_countries)
        except Exception:
            continue
        
    # Remove elements in fine_grained_answers that are also in possible_base_answers and current adversarial answers
    coarse_grained_answers = coarse_grained_answers - set(possible_base_answers)
    
    adversarial_countries = fine_grained_answers.union(coarse_grained_answers)
    if len(adversarial_countries) == 0:
        # Get remaining answers in adversarial answers (random) countries
        remain_answers = set()
        remain_foods = food_raw_df[food_raw_df['food_id'].isin(third_indices)].reset_index(drop=True)
        for _, remain_row in remain_foods.iterrows():
            remain_countries = remain_row['Countries']
            try:
                remain_countries = ast.literal_eval(remain_countries)
                remain_answers.update(remain_countries)
            except Exception:
                continue
        remain_answers = remain_answers - set(possible_base_answers)
        adversarial_countries = adversarial_countries.union(remain_answers)
        
    adversarial_countries = list(adversarial_countries)
    indexing = rnd_int
    if len(adversarial_countries) != 0 and adver_or_not < ODDS_ADVERSARIAL:
        # Ensure it's in base key, otherwise pick other
        potential_base_key = adversarial_countries[indexing % len(adversarial_countries)]
        while indexing < rnd_int + len(adversarial_countries):
            filtered_loc = location_cuisine_df[location_cuisine_df['base_key'] == potential_base_key]
            if len(filtered_loc) > 0:
                return potential_base_key
            else:
                indexing += 1
        
    # Reaches here means either base key is not there for adversarial countries or simply winning the odds
    indexing = rnd_int
    potential_base_key = possible_base_answers[indexing % len(possible_base_answers)]
    while indexing < rnd_int + len(possible_base_answers):
        filtered_loc = location_cuisine_df[location_cuisine_df['base_key'] == potential_base_key]
        if len(filtered_loc) > 0:
            return potential_base_key
        else:
            indexing += 1
            
    # Shouldn't reach here, should we generate random countries instead?
    raise RuntimeError("For some reason can't find countries from prompt")
        

def generate_prompt(prompt, language, food_row, location_cuisine_df, rnd_int, adver_or_not):
    prompts = ""
    matches = re.findall(r'<(.*?)>', prompt)
    country_list = ast.literal_eval(food_row['Countries'])
    actual_language_name = LANGUAGE_CODE_MAPPING[language]
    
    if food_row['prompt_type'] != 4: # If not adversarial type
        if food_row['answer'] in country_list:
            base_key = food_row['answer']
        else:
            base_key = country_list[0]
    else:
        # Get random location that is not in country_list
        base_key = generate_adversarial_loc_cuis_for_prompt(food_row, rnd_int, adver_or_not, location_cuisine_df)

    if len(matches) == 0:
        return prompt  # No placeholders to replace

    for key_match in matches:
        try:
            if 'LOCATION' in key_match or 'CUISINE' in key_match:
                if language == "es" and key_match == "CUISINE":
                    key_match = "CUISINE.f"
                if key_match == "LOCATION_PEOPLE":
                    key_match = "LOCATION"
                to_change = f"<{key_match}>"
                
                filtered_df = location_cuisine_df[location_cuisine_df['base_key'] == base_key]
                change_to = filtered_df[f"{actual_language_name.split('_')[0]}_{to_change}"].values[0] if not filtered_df.empty else None
                
                if change_to is not None:
                    prompts = prompt.replace(to_change, change_to)
                    break

        except Exception as e:
            logging.warning(f"Error {e}")
            logging.warning(f"prompt = {prompt}, to_change = {to_change}, language = {language}, key_match = {key_match}, base_key = {base_key}")
            continue
        
    return prompts

def apply_generate_prompt(row, lang, location_cuisine_df, rnd_int, adver_or_not):
    prompt = row[LANGUAGE_CODE_MAPPING[lang]]
    food_row = row  # Treat the whole row as food_row
    return generate_prompt(prompt, lang, food_row, location_cuisine_df, rnd_int, adver_or_not)

def generate_prompt_dataset(sampled_df, food_cleaned_df, qc_df, location_cuisine_df, languages_used):
    random_int_for_adversarial_indexing = np.random.randint(0, 1000, size=1)[0]
    random_float = np.random.uniform(0, 1, size=1)[0]
    temp_df = pd.merge(sampled_df, food_cleaned_df[['food_id', f"Top_{TOP_K}_Same_Fine_Category",
                                                    f'Top_{TOP_K}_Same_Coarse_Category',
                                                    f'Top_{TOP_K}_Similar_Foods']], how='left', on='food_id')
    temp_df = pd.merge(temp_df[["prompt_id", "answer", 'Countries',
                                   f"Top_{TOP_K}_Same_Fine_Category", f'Top_{TOP_K}_Same_Coarse_Category',
                                   f'Top_{TOP_K}_Similar_Foods']], qc_df, how='left', on=['prompt_id'])
    for language in languages_used:
        sampled_df[f"prompt_{language}"] = temp_df.apply(apply_generate_prompt, axis=1, lang=language, location_cuisine_df=location_cuisine_df,
                                                         rnd_int=random_int_for_adversarial_indexing, adver_or_not=random_float)
        
    return sampled_df
        
# need_sample is legacy, now we sample w.r.t to prompt for each dish
def generate_initial_combinations(current_dataset, all_combinations_df, cleaned_food_df,
                                  n_prompt_max_type1, n_prompt_max_type2,
                                  n_prompt_max_type3, n_prompt_max_type4,
                                  need_sample, seed):
    start_time = time.time()
    # Create all possible combinations of 'food_id' and 'prompt_id' using a cross join (Cartesian product)
    sampled_dataset_keys = pd.DataFrame(columns=['food_id', 'prompt_id'])
    if len(current_dataset) != 0:
        sampled_dataset_keys = current_dataset[['food_id', 'prompt_id']].drop_duplicates()
    
    # Legacy code for sampling overall
    # If the number of samples requested is larger than available combinations, limit it
    # if need_sample <= 0 or need_sample > len(all_combinations_df):
    #     need_sample = len(all_combinations_df)
    # filtered_sampled_df = all_combinations_df.sample(n=need_sample, random_state=seed).reset_index(drop=True)
    
    # Initialize an empty list to store the results
    result_list = []
    
    for food_id, group_df in all_combinations_df.groupby('food_id'):
        # Sample for prompt_id type 1, 2, 3, 4
        if n_prompt_max_type1 >= 0:
            type1_sample = group_df[group_df['prompt_type'] == 1].sample(n=n_prompt_max_type1, random_state=seed, replace=False)
        else:
            type1_sample = group_df[group_df['prompt_type'] == 1]
        
        if n_prompt_max_type2 >= 0:
            type2_sample = group_df[group_df['prompt_type'] == 2].sample(n=n_prompt_max_type2, random_state=seed, replace=False)
        else:
            type2_sample = group_df[group_df['prompt_type'] == 2]
            
        if n_prompt_max_type3 >= 0:
            type3_sample = group_df[group_df['prompt_type'] == 3].sample(n=n_prompt_max_type3, random_state=seed, replace=False)
        else:
            type3_sample = group_df[group_df['prompt_type'] == 3]
            
        if n_prompt_max_type4 >= 0:
            type4_sample = group_df[group_df['prompt_type'] == 4].sample(n=n_prompt_max_type4, random_state=seed, replace=False)
        else:
            type4_sample = group_df[group_df['prompt_type'] == 4]
        
        # Append all the samples for this food_id to the result list
        result_list.append(type1_sample)
        result_list.append(type2_sample)
        result_list.append(type3_sample)
        result_list.append(type4_sample)

    # Combine the results into a single DataFrame
    filtered_sampled_df = pd.concat(result_list).reset_index(drop=True)
    
    # Merge on the three columns and keep only rows combinations that are not in current_dataset
    filtered_sampled_df = pd.merge(filtered_sampled_df, sampled_dataset_keys, on=['food_id', 'prompt_id'], 
                                        how='left', indicator=True)

    # Keep only the rows that are not present in current_dataset
    filtered_sampled_df = filtered_sampled_df[filtered_sampled_df['_merge'] == 'left_only'].drop(columns='_merge')

    # If Answer is Location, then sample different locations
    sampled_answers = []
    for _, row in filtered_sampled_df.iterrows():
        # Sample answer if it questions about Location
        sampled_answer = sample_an_answer(row, cleaned_food_df)
        sampled_answers.append(sampled_answer)
    filtered_sampled_df['answer'] = sampled_answers
    
    end_time = time.time()
    logging.debug(f"Time takes to generate initial combinations is {end_time - start_time:.3f} seconds, resulting in {len(filtered_sampled_df)} rows")
    
    return filtered_sampled_df

def sample_dataset(food_raw_df, food_cleaned_df, query_context_df, location_cuisine_df, languages_used,
                   n=20000, alias_aware=False,
                   n_dish_max=-1, n_prompt_max_type1=-1, n_prompt_max_type2=-1, n_prompt_max_type3=-1, n_prompt_max_type4=-1,
                   is_eval=True, max_retry=3):
    """
    1. Get all possible combinations based on n_dish_max, n_prompt_max_type1,2,3,4
    2. Sample possible food_id, prompt_id, then proceed to sample the image indices
    3. Get the (parallel) prompts based on the language input used
    4. Sample challenging/adversarial answers
    """
    sampled_dataset = pd.DataFrame()
    curr_seed = RANDOM_SEED
    num_sampled_rows = 0
    
    # Calculate total combinations
    global_sampled_food_df, global_sampled_qc_df = eval_test_possible_combinations(food_cleaned_df, query_context_df, is_eval)
    image_url_counts = [len(image_urls) for image_urls in global_sampled_food_df['ImageUrls']]
    total_image_urls = sum(image_url_counts)
    total_prompts = len(global_sampled_qc_df['prompt_id'])
    num_total_combinations = total_image_urls * total_prompts
    all_combinations_df = filtered_combinations(global_sampled_food_df, global_sampled_qc_df, food_cleaned_df,
                                                location_cuisine_df, n_dish_max)
    total_to_be_sampled = ceil(n / len(languages_used)) # n here represents total with all languages

    retry = 0
    while num_sampled_rows < total_to_be_sampled:
        logging.debug(f"Current sampled rows: {num_sampled_rows}")
        curr_seed += 1
        need_sample = min(total_to_be_sampled - num_sampled_rows, num_total_combinations)
        
        filtered_sampled_df = generate_initial_combinations(sampled_dataset, all_combinations_df, food_cleaned_df,
                                                            n_prompt_max_type1, n_prompt_max_type2,
                                                            n_prompt_max_type3, n_prompt_max_type4,
                                                            need_sample, curr_seed)

        if len(filtered_sampled_df) == 0 and retry < max_retry:
            retry += 1
            logging.info("All combinations have been exhausted, attempting to resample!")
            continue
        elif len(filtered_sampled_df) == 0 and retry == max_retry:
            logging.info("All combinations have been exhausted, and max retry has been reached, stop sampling!")
            break
        
        start_time = time.time()
        
        # Generate prompts for each languages
        prompted_sampled_df = generate_prompt_dataset(filtered_sampled_df, food_cleaned_df,
                                                      query_context_df, location_cuisine_df, languages_used)
        
        # Get challenging answers to the results
        further_filt_sampled_df = get_nearest_answers(prompted_sampled_df,
                                                food_raw_df,
                                                food_cleaned_df,
                                                location_cuisine_df, languages_used,
                                                alias_aware)
        further_filt_sampled_df = further_filt_sampled_df.dropna()
        
        # Concat
        further_filt_sampled_df = further_filt_sampled_df.sample(frac=1, random_state=RANDOM_SEED)
        if (len(further_filt_sampled_df) + num_sampled_rows < total_to_be_sampled):
            sampled_dataset = pd.concat([sampled_dataset, further_filt_sampled_df], ignore_index=True)
            num_sampled_rows += len(further_filt_sampled_df)
        else:
            take_few_rows = total_to_be_sampled - num_sampled_rows
            sampled_dataset = pd.concat([sampled_dataset, further_filt_sampled_df[:take_few_rows]], ignore_index=True)
            num_sampled_rows += take_few_rows

        end_time = time.time()
        logging.debug(f"Time takes to filter combinations is {end_time - start_time:.3f} seconds, resulting in {len(further_filt_sampled_df)} rows")

    # Drop action as it was originally used for getting answer type
    sampled_dataset = sampled_dataset.sample(frac=1, random_state=RANDOM_SEED) # Final shuffle
    sampled_dataset = sampled_dataset.drop(columns=['action'])
    
    return sampled_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # TODO: Take argument as json file maybe so it's easier?
    parser.add_argument('-o', '--output_csv', type=str, required=True, help="CSV path where sampled data will be saved.")
    parser.add_argument('-fr', '--food_raw_path', type=str, required=False, default=os.path.join(RESOURCE_DIR, "food_raw_6oct.csv"), help="Path used to load raw food dataframe")
    parser.add_argument('-fc', '--food_cleaned_path', type=str, required=False, default=os.path.join(RESOURCE_DIR, "food_cleaned.csv"), help="Path used to load cleaned food dataframe")
    parser.add_argument('-q', '--query_context_path', type=str, required=False, default=os.path.join(RESOURCE_DIR, "query_ctx.csv"), help="Path used to load the query context CSV")
    parser.add_argument('-l', '--loc_cuis_path', type=str, required=False, default=os.path.join(RESOURCE_DIR, "location_and_cuisine.csv"), help="Path used to load the use location/cuisine CSV")
    parser.add_argument('-n', '--num_samples', type=int, required=False, default=20000, help="Number of samples (multiply by the languages)")
    parser.add_argument('-ll', '--list_of_languages', type=str, required=False, default="", help="List of languages used (e.g. '[\"en\", \"id_formal\"]')")
    parser.add_argument('-aw', '--alias_aware', default=False, action=argparse.BooleanOptionalAction, help="Enabling this will have the sampler tries harder to get the adversarial answers with parallel Aliases")
    parser.add_argument('-nd', '--n_dish_max', type=int, required=False, default=-1, help="Maximum different number of dishes to be sampled (-1 means can sample as many as possible).")
    parser.add_argument('-np1', '--n_prompt_max_type1', type=int, required=False, default=-1, help="Maximum different number of prompt type 1 to be sampled (-1 means can sample as many as possible).")
    parser.add_argument('-np2', '--n_prompt_max_type2', type=int, required=False, default=-1, help="Maximum different number of prompt type 2 to be sampled (-1 means can sample as many as possible).")
    parser.add_argument('-np3', '--n_prompt_max_type3', type=int, required=False, default=-1, help="Maximum different number of prompt type 3 to be sampled (-1 means can sample as many as possible).")
    parser.add_argument('-np4', '--n_prompt_max_type4', type=int, required=False, default=-1, help="Maximum different number of prompt type 4 to be sampled (-1 means can sample as many as possible).")
    parser.add_argument('-ie', '--is_eval', default=True, action=argparse.BooleanOptionalAction, help="Whether train or evaluation.")
    args = parser.parse_args()
    
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Cannot have all max_type to be 0
    assert(not (args.n_prompt_max_type1 == 0 and args.n_prompt_max_type2 == 0 and args.n_prompt_max_type3 == 0 and args.n_prompt_max_type4 == 0))
    
    os.makedirs(RESOURCE_DIR, exist_ok=True)
    
    food_raw_df = pd.read_csv(args.food_raw_path)
    food_cleaned_df = pd.read_csv(args.food_cleaned_path)
    query_context_df = pd.read_csv(args.query_context_path)
    location_cuisine_df= pd.read_csv(args.loc_cuis_path)
    
    list_of_languages = ALL_LANGUAGES
    if args.list_of_languages != "":
        list_of_languages = ast.literal_eval(args.list_of_languages)  

    start_time = time.time()

    samples = sample_dataset(food_raw_df, food_cleaned_df, query_context_df, location_cuisine_df, list_of_languages,
                             n=args.num_samples, alias_aware=args.alias_aware, n_dish_max=args.n_dish_max,
                             n_prompt_max_type1=args.n_prompt_max_type1, n_prompt_max_type2=args.n_prompt_max_type2,
                             n_prompt_max_type3=args.n_prompt_max_type3, n_prompt_max_type4=args.n_prompt_max_type4,
                             is_eval=args.is_eval)

    end_time = time.time()
    
    logging.info(f"Time takes to sample {args.num_samples} rows is {end_time - start_time:.3f} seconds")
    
    samples.to_csv(args.output_csv, index=False)