import random
import ast
import pandas as pd
import numpy as np
import argparse

from utils import *

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
        return row['Answer']
    else:
        food_id = row['food_id']
        countries = food_df.loc[food_df['food_id'] == food_id, 'Countries'].values[0]
        try:
            countries = ast.literal_eval(countries)  
            # Sample random countries
            country = countries[random.randint(0, len(countries) - 1)]
            return country
        except Exception:
            # Wait... how did we get the Answer then? Hmmm
            return row['Answer']


def sample_rows(filtered_full_df, sampled_dataset, food_df, need_sample, num_iter):
    new_sampled_df = filtered_full_df.copy(True)
    
    # Shuffle the DataFrame and take first need_sample rows if not -1; num_iter as random seed so shuffling is psuedo-random 
    new_sampled_df = new_sampled_df.sample(frac=1, random_state=num_iter).reset_index(drop=True)
    
    # In case DataFrame is empty (initial sampling)
    sampled_dataset_keys = pd.DataFrame(columns=['food_id', 'prompt_id', 'Image Index'])
    if len(sampled_dataset) != 0:
        sampled_dataset_keys = sampled_dataset[['food_id', 'prompt_id', 'Image Index']].drop_duplicates()

    # Iterate over shuffled_df to sample an image for each row based on the food_id
    # Additionally if Answer is Location, then sample different locations
    sampled_image_indices = []
    sampled_images = []
    sampled_answers = []
    for _, row in new_sampled_df.iterrows():
        food_id = row['food_id']
        
        # Sample the image
        sampled_index, sampled_image_url = sample_an_image(food_id, food_df)
        sampled_image_indices.append(sampled_index)
        sampled_images.append(sampled_image_url)
        
        # Sample answer if it questions about Location
        sampled_answer = sample_an_answer(row, food_df)
        sampled_answers.append(sampled_answer)
    new_sampled_df['Image'] = sampled_images
    new_sampled_df['Image Index'] = sampled_image_indices
    new_sampled_df['Answer'] = sampled_answers
    
    # Merge on the three columns and keep only the rows from new_sampled_df that are not in sampled_dataset
    filtered_sampled_df = new_sampled_df.merge(sampled_dataset_keys, on=['food_id', 'prompt_id', 'Image Index'], 
                                        how='left', indicator=True)

    # Keep only the rows that are not present in sampled_dataset
    filtered_sampled_df = filtered_sampled_df[filtered_sampled_df['_merge'] == 'left_only'].drop(columns='_merge')
        
    if need_sample != -1:
        filtered_sampled_df = filtered_sampled_df.head(need_sample)
    
    return filtered_sampled_df

def get_food_indices_given_row(row):
    # Get indices for the nearest answers
    index_map = {
        'Top_15_Same_Fine_Category': row['Top_15_Same_Fine_Category'].values[0],
        'Top_15_Same_Coarse_Category': row['Top_15_Same_Coarse_Category'].values[0],
        'Top_15_Similar_Foods': row['Top_15_Similar_Foods'].values[0]
    }
    try:
        index_map['Top_15_Same_Fine_Category'] = ast.literal_eval(index_map['Top_15_Same_Fine_Category'])
    except Exception:
        index_map['Top_15_Same_Fine_Category'] = []
        
    try:
        index_map['Top_15_Same_Coarse_Category'] = ast.literal_eval(index_map['Top_15_Same_Coarse_Category'])
    except Exception:
        index_map['Top_15_Same_Coarse_Category'] = []


    try:
        index_map['Top_15_Same_Similar_Foods'] = ast.literal_eval(index_map['Top_15_Similar_Foods'])
    except Exception:
        index_map['Top_15_Same_Similar_Foods'] = []
        
    return index_map

def shuffle_answers_with_correct_index(wrong_answers, correct_answer):
    shuffled_answers = wrong_answers.copy()
    correct_index = random.randint(0, len(shuffled_answers))
    shuffled_answers.insert(correct_index, correct_answer)
    return shuffled_answers, correct_index

def generate_translate_food_alias_aware(row, food_df, all_indices_sets, languages_used):
    first_indices = all_indices_sets[0]
    second_indices = all_indices_sets[1]
    third_indices = all_indices_sets[2]
    
    adversarial_answers = []

    # Find indices that contain all aliases for each level
    for indices in all_indices_sets:
        for index in indices:
            # Assuming 'Alias' is a string representation of a dictionary
            alias_dict = ast.literal_eval(food_df.loc[index, 'Alias'])
            alias_dict = {lang.split(' - ')[-1]: alias_dict[lang] for lang in alias_dict}

            for lang in languages_used:
                translation = food_df.loc[index, 'Name']
                no_alias = True
                for alias_lang in ALIAS_MAPPING[lang]:
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
    adversarial_answers = [food_df.loc[idx, 'Name'] for idx in adversarial_answers]
    shuffled_answers, answer_index = shuffle_answers_with_correct_index(adversarial_answers, row['Answer'])
    
    # Convert each answer into each language name using aliasing
    translation_dict = {}
    lang_status = 'full'
    for ans in shuffled_answers:
        index = food_df[food_df['Name'].str.strip() == ans.strip()].index[0]
        # Assuming 'Alias' is a string representation of a dictionary
        alias_dict = ast.literal_eval(food_df.loc[index, 'Alias'])
        alias_dict = {lang.split(' - ')[-1]: alias_dict[lang] for lang in alias_dict}

        for lang in languages_used:
            translation = food_df.loc[index, 'Name']
            no_alias = True
            for alias_lang in ALIAS_MAPPING[lang]:
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

def generate_translate_food_default(row, food_df, all_indices_sets, languages_used):
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
    adversarial_answers = [food_df.loc[idx, 'Name'] for idx in adversarial_answers]
    shuffled_answers, answer_index = shuffle_answers_with_correct_index(adversarial_answers, row['Answer'])
    
    # Convert each answer into each language name using aliasing
    translation_dict = {}
    lang_status = 'full'
    for ans in shuffled_answers:
        index = food_df[food_df['Name'].str.strip() == ans.strip()].index[0]
        # Assuming 'Alias' is a string representation of a dictionary
        alias_dict = ast.literal_eval(food_df.loc[index, 'Alias'])
        alias_dict = {lang.split(' - ')[-1]: alias_dict[lang] for lang in alias_dict}

        for lang in languages_used:
            translation = food_df.loc[index, 'Name']
            no_alias = True
            for alias_lang in ALIAS_MAPPING[lang]:
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

def generate_location_base_answers(row, food_df, all_indices_sets):
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
        cur_answers = row['Location']
        cur_answers = ast.literal_eval(cur_answers)
        possible_base_answers.extend(cur_answers)
    except Exception:
        pass
    
    adversarial_answers = []
    
    # Get 2 answers from (random) countries in fine-grained
    fine_grained_answers = set()
    fine_foods = food_df[food_df['food_id'].isin(first_indices)].reset_index(drop=True)
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
    coarse_foods = food_df[food_df['food_id'].isin(second_indices)].reset_index(drop=True)
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
    remain_foods = food_df[food_df['food_id'].isin(third_indices)].reset_index(drop=True)
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
        
    if len(adversarial_answers) != MAX_ANS_ALL:
        logging.warning(f"For some reason number of adversarial answers for location is not {MAX_ANS_ALL}")
        return [row['Answer']], 0

    # Return only the required number of answers
    shuffled_answers, answer_index = shuffle_answers_with_correct_index(adversarial_answers, row['Answer'])
    
    return shuffled_answers, answer_index

def translate_location_languages(all_answers, location_cuisine_df, languages_used):
    translation_dict = {}
    for ans in all_answers:
        index = location_cuisine_df[location_cuisine_df['English_<LOCATION>'].str.strip() == ans.strip()].index[0]

        # Create a dictionary for the translations
        translations = {
            lang: location_cuisine_df.loc[index, LOCATION_MAPPING[lang]] 
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
    
def generate_answers_per_language(row, food_df, location_cuisine_df,
                                  all_indices_sets, languages_used, alias_aware=False):
    if row['action'] == 'Name':
        # If the question is about food name, cautious about aliases
        if alias_aware:
            translation_dict, correct_index, lang_status = generate_translate_food_alias_aware(row, food_df, all_indices_sets, languages_used)
        else:
            translation_dict, correct_index, lang_status = generate_translate_food_default(row, food_df, all_indices_sets, languages_used)
        return translation_dict, correct_index, lang_status
    elif row['action'] == 'Location':
        # Otherwise, pretty sure other languages will have the translation
        all_answers, correct_index = generate_location_base_answers(row, food_df, all_indices_sets)
        if len(all_answers) != (MAX_ANS_ALL + 1):
            return {}, -1, "failed"
        translation_dict = translate_location_languages(all_answers, location_cuisine_df, languages_used)
        return translation_dict, correct_index, "full" # Assume always successful translation
    else:
        raise NotImplementedError(f"Answer type {row['action']} has not yet been implemented")

def get_nearest_answers(filtered_sampled_df, food_df, location_cuisine_df, languages_used, alias_aware=False):
    # Iterate through each row in the subset_df
    for idx, row in filtered_sampled_df.iterrows():
        food_id = row['food_id']
        food_row = food_df.loc[food_df['food_id'] == food_id]
        index_map = get_food_indices_given_row(food_row)

        all_indices_sets = []

        # Create the first set from the first list
        first_indices = set(index_map.get('Top_15_Same_Fine_Category', []))
        all_indices_sets.append(first_indices)

        # Create the second set from the second list while removing duplicates from the first
        second_indices = set(index_map.get('Top_15_Same_Coarse_Category', [])) - first_indices
        all_indices_sets.append(second_indices)

        # Create the third set from the third list while removing duplicates from the first and second
        third_indices = set(index_map.get('Top_15_Same_Similar_Foods', [])) - first_indices - second_indices
        all_indices_sets.append(third_indices)
        
        # Get answers per languages
        answers_dict, correct_index, lang_status = generate_answers_per_language(row, food_df, location_cuisine_df,
                                                              all_indices_sets, languages_used, alias_aware)
        for lang, answers in answers_dict.items():
            filtered_sampled_df.at[idx, f'Answer_{lang}'] = str(answers)
        filtered_sampled_df.at[idx, 'correct_index'] = correct_index
        filtered_sampled_df.at[idx, 'lang_status'] = lang_status
        
    further_filt_sampled_df = filtered_sampled_df[filtered_sampled_df['lang_status'] != 'failed']
    
    return further_filt_sampled_df
    
def filter_full_dataset(full_df, query_context_df, food_df, location_cuisine_df):
    # Drop rows in full_df where it contains invalid prompt
    query_context_df['action'] = query_context_df['Template'].map(CONTEXT_TYPE_ACTION)
    drop_rows = query_context_df[query_context_df['action'] == '<DROP>']
    filtered_full_df = full_df[~full_df['prompt_id'].isin(drop_rows['prompt_id'])]
    action_df = query_context_df[['prompt_id', 'action']]
    filtered_full_df = pd.merge(filtered_full_df, action_df, on='prompt_id', how='left') # Get action_df for later
    
    # Filter dataset where location is not proper (not in location_cuisine_df)
    standardized_locations = location_cuisine_df['English_<LOCATION>'].to_list()
    standardized_locations = [location.strip() for location in standardized_locations]
    if 'global' in standardized_locations:
        standardized_locations.remove('global')
    if 'Global' in standardized_locations:
        standardized_locations.remove('Global') 
    
    # First, join filtered_full_df with food_df to bring in the Countries column
    # Assuming food_id is the key for joining
    merged_df = pd.merge(filtered_full_df, food_df[['food_id', 'Countries']], on='food_id', how='left')
    merged_df['Countries'] = merged_df['Countries'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Function to filter and reduce countries
    def filter_and_reduce_countries(row):
        countries = row['Countries']
        # Filter countries to only keep those that are in standardized_locations
        reduced_countries = [country for country in countries if country in standardized_locations]
        # Check if reduced countries is not empty
        return reduced_countries if reduced_countries else None  # Or return an empty list if you prefer

    # Apply the filtering and reducing function
    merged_df['Countries'] = merged_df.apply(filter_and_reduce_countries, axis=1)

    # Remove rows where 'Countries' is None or an empty list
    filtered_full_df = merged_df[merged_df['Countries'].apply(lambda x: x is not None and len(x) > 0)]
    filtered_full_df = filtered_full_df.drop(columns=['Countries'])
    # How about if the answer is still wrong? It may, but no worries, 
    # we will do answer sampling later anyway, so any placeholder is just as fine~

    return filtered_full_df
    

def sample_dataset(full_df, query_context_df, food_df, location_cuisine_df, n=20000, sample_batch=-1, max_attempts=3, alias_aware=False):
    sampled_dataset = pd.DataFrame()
    
    filtered_full_df = filter_full_dataset(full_df, query_context_df, food_df, location_cuisine_df)

    # Get languages based on prompt
    languages_avail = filtered_full_df.columns
    languages_used = [col.replace('Prompt_', '') for col in languages_avail if col.startswith('Prompt_')]
    
    num_iter = 0
    num_sampled_rows = 0
    while num_sampled_rows < n:
        logging.debug(f"Current sampled rows: {num_sampled_rows}")
        num_iter += 1
        attempts = 0
        while attempts < max_attempts:
            need_sample = min(n - num_sampled_rows, len(filtered_full_df))
            if sample_batch != -1:
                need_sample = min(need_sample, sample_batch)
            
            filtered_sampled_df = sample_rows(filtered_full_df, sampled_dataset, food_df, need_sample, num_iter)
            
            if len(filtered_sampled_df) == 0:
                attempts += 1
                continue
            
            # Get hallenging answers to the results
            further_filt_sampled_df = get_nearest_answers(filtered_sampled_df,
                                                 food_df,
                                                 location_cuisine_df, languages_used,
                                                 alias_aware)
            
            # Concat
            sampled_dataset = pd.concat([sampled_dataset, further_filt_sampled_df], ignore_index=True)
            num_sampled_rows += len(further_filt_sampled_df)

            break
        else:
            logging.warning(f"Warning: Ran out of valid samples after {max_attempts} attempts.")
            break
    
    # Drop action as it was originally used for getting answer type
    sampled_dataset = sampled_dataset.drop(columns=['action'])
    
    return sampled_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #full_df, query_context_df, food_df, location_cuisine_df, n=200000, sample_batch=-1, max_attempts=3, alias_aware=False
    # Arguments for experiment regressor mode
    parser.add_argument('-f', '--full_df_path', type=str, required=False, default=os.path.join(RESOURCE_DIR, "full_df.csv"), help="Path used to load full dataframe")
    parser.add_argument('-o', '--food_df_path', type=str, required=False, default=os.path.join(RESOURCE_DIR, "food_df.csv"), help="Path used to load food dataframe")
    parser.add_argument('-q', '--query_context_path', type=str, required=False, default=os.path.join(RESOURCE_DIR, "query_ctx.csv"), help="Path used to load the query context CSV")
    parser.add_argument('-l', '--loc_cuis_path', type=str, required=False, default=os.path.join(RESOURCE_DIR, "loc_cuis.csv"), help="Path used to load the use location/cuisine CSV")
    parser.add_argument('-n', '--num_samples', type=int, required=False, default=20000, help="Number of samples")
    parser.add_argument('-m', '--max_attempts', type=int, required=False, default=3, help="Maximum number of attempts resampling in case new subset cannot be found")
    args = parser.parse_args()
    
    os.makedirs(RESOURCE_DIR, exist_ok=True)
    
    # full_df = pd.read_csv(args.full_df_path)
    # food_df = pd.read_csv(args.food_df_path)
    # query_context_df = pd.read_csv(args.query_context_path)
    # location_cuisine_dict = pd.read_csv(args.loc_suic_path)
    
    from datasets import load_dataset
    import pandas as pd
    import numpy as np
    from huggingface_hub import login

    # Log in with your API key
    login("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")
    
    # Load your private dataset by its name or path in your account
    dataset = load_dataset("Exqrch/cuisine-location_and_cuisine")
    location_cuisine_df = pd.DataFrame(dataset['train'])

    dataset = load_dataset("Exqrch/cuisine-query_and_context")
    query_context_df = pd.DataFrame(dataset['train'])

    dataset = load_dataset("Exqrch/cuisine-food")
    food_df = pd.DataFrame(dataset['train'])

    # Took 1 min
    dataset = load_dataset("Exqrch/cuisine-v1")
    full_df = pd.DataFrame(dataset['train'])
 
    
    sample_dataset(full_df, query_context_df, food_df, location_cuisine_df, n=args.num_samples)
