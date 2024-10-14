import pandas as pd
import json
import re
import yaml

from datasets import load_dataset
from tqdm import tqdm
from collections import OrderedDict
from evaluate import load

with open('score.yml', 'r') as file:
    CONFIG = yaml.safe_load(file)

# src files
TASK1_MC_PATH = "../result/task1_mc_{model}_pred.jsonl"
TASK1_OE_PATH = "../result/task1_oe_{model}_pred.jsonl"
TASK2_MC_PATH = "../result/task2_mc_{model}_pred.jsonl"
TASK2_OE_PATH = "../result/task2_oe_{model}_pred.jsonl"

# target files
ACCURACY_MC_PATH = "./json/{model}_accuracy_mc.json"
ACCURACY_OE_PATH = "./json/{model}_accuracy_oe.json"
BERTSCORE_OE_PATH = "./json/{model}_bertscore_oe.json"
ERROR_MC_PATH = "./error/{model}_error_mc.txt"

if cfg["subset"] == "small":
    # src files (small subset)
    TASK1_MC_PATH = "../result/small/task1_mc_{model}_pred_small.jsonl"
    TASK1_OE_PATH = "../result/small/task1_oe_{model}_pred_small.jsonl"
    TASK2_MC_PATH = "../result/small/task2_mc_{model}_pred_small.jsonl"
    TASK2_OE_PATH = "../result/small/task2_oe_{model}_pred_small.jsonl"

    # target files (small subset)
    ACCURACY_MC_PATH = "./json/small/{model}_accuracy_mc.json"
    ACCURACY_OE_PATH = "./json/small/{model}_accuracy_oe.json"
    BERTSCORE_OE_PATH = "./json/small/{model}_bertscore_oe.json"
    ERROR_MC_PATH = "./error/small/{model}_error_mc.txt"


def load_jsonl(file_path):
  data = []
  with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
      data.append(json.loads(line))
  return data

def load_result(model, mode = "all"):
    res = []
    if mode == "all" or mode == "mc":
        task1_mc = pd.DataFrame(load_jsonl(TASK1_MC_PATH.format(model=model)))
        task2_mc = pd.DataFrame(load_jsonl(TASK2_MC_PATH.format(model=model)))
        res.extend([task1_mc, task2_mc])
    
    if mode == "all" or mode == "oe":
        task1_oe = pd.DataFrame(load_jsonl(TASK1_OE_PATH.format(model=model)))
        task2_oe = pd.DataFrame(load_jsonl(TASK2_OE_PATH.format(model=model)))
        res.extend([task1_oe, task2_oe])

    vqa_task1 = load_dataset("world-cuisines/vqa", "task1", split = "test_large").to_pandas()
    vqa_task2 = load_dataset("world-cuisines/vqa", "task2", split = "test_large").to_pandas()

    qa_id_prompt_type_dict = vqa_task1.set_index('qa_id')['prompt_type'].to_dict()
    qa_id_prompt_type_dict.update(vqa_task2.set_index('qa_id')['prompt_type'].to_dict())

    df_res = pd.concat(res, ignore_index = True)
    df_res['qa_id'] = df_res['qa_id'].astype(str)
    df_res['lang'] = df_res['lang'].fillna("nan")
    df_res['prompt_type'] = df_res.qa_id.map(qa_id_prompt_type_dict)

    df_res['type'] = 'oe'
    df_res.loc[df_res.multi_choice_answer.notna(), 'type'] = 'mc'
    df_res.loc[df_res.multi_choice_answer.notna(), 'answer'] = df_res.loc[df_res.multi_choice_answer.notna(), 'multi_choice_answer']
    df_res = df_res.drop(columns=['multi_choice_answer'])

    return df_res, vqa_task1, vqa_task2

def final_cat_mapping(scores):
    for category in scores:
        if isinstance(scores[category], dict):
            if 'su_loma' in scores[category]:
                scores[category]['su'] = scores[category].pop('su_loma')
            if 'si_formal_spoken' in scores[category]:
                scores[category]['si'] = scores[category].pop('si_formal_spoken')

            scores[category] = OrderedDict(sorted(scores[category].items()))

        mapping = {
            "1": "1a. Dish Name (No-Context)",
            "2": "2. Regional Cuisine",
            "3": "1b. Dish Name (Contextualized)",
            "4": "1c. Dish Name (Adversarial)"
        }
        
        mapped_scores = {mapping.get(k, k): v for k, v in scores.items()}

    return mapped_scores


def score_mc(model, df_res, vqa_task1, vqa_task2):
    errs = []

    for index, row in tqdm(df_res[df_res['type'] == 'mc'].iterrows(), total=len(df_res[df_res['type'] == 'mc'])):
        # Check if answer is number only
        try:
            prediction = row['prediction']
            df_res.loc[index, 'prediction'] = int(prediction)
        except ValueError:
            # Regex pattern to match "<int>." at the beginning 
            pattern = r"^(\d+)\."
            try:
                matched = re.match(pattern, str(prediction))
                if matched:
                    extracted_int = int(matched.group(1))
                    df_res.loc[index, 'prediction'] = extracted_int
                else:
                    raise ValueError(f"No integer found in the beginning of prediction: {prediction}")
            except ValueError:
                # Regex pattern to match any number
                try:
                    # Regex pattern to match any "<int>"
                    pattern = r"(\d+)"
                    matched = re.search(pattern, str(prediction))
                    if matched:
                        extracted_int = int(matched.group(0))
                        df_res.loc[index, 'prediction'] = extracted_int
                    else:
                        raise ValueError(f"No integer found in prediction: {prediction}")
                except ValueError:
                    # check if answer is one of the multiple choices, even though there are no number
                    try:
                        if row['prompt_type'] == '2':
                            vqa_row = vqa_task2[(vqa_task2['qa_id'] == row['qa_id']) & (vqa_task2['lang'] == row['lang'])]
                            prompt = vqa_row['multi_choice_prompt'].iloc[0]
                            prediction = prediction.replace('.', '')
                            matched = prompt.find(prediction)
                            if matched != -1:
                                df_res.loc[index, 'prediction'] = prompt[matched - 3]
                            else:
                                raise ValueError(f"No answer found in prediction: {prediction}")
                        else:
                            vqa_row = vqa_task1[(vqa_task1['qa_id'] == row['qa_id']) & (vqa_task1['lang'] == row['lang']) & (vqa_task1['prompt_type'] == row['prompt_type'])]
                            prompt = vqa_row['multi_choice_prompt'].iloc[0]
                            prediction = prediction.replace('.', '')
                            matched = prompt.find(prediction)
                            if matched != -1:
                                df_res.loc[index, 'prediction'] = prompt[matched - 3]
                            else:
                                raise ValueError(f"No answer found in prediction: {prediction}")
                    except Exception as e:
                        error_msg = f"Row {index}: {row['prediction']} - {e}\n"
                        errs.append(error_msg)

    accuracies = {}
    n = 0
    for prompt_type in tqdm(df_res['prompt_type'].unique(), total = len(df_res['prompt_type'].unique())):
        accuracies[prompt_type] = {} 
        for lang in tqdm(df_res['lang'].unique(), total = len(df_res['lang'].unique())):
            df_subset = df_res[(df_res['type'] == 'mc') & (df_res['prompt_type'] == prompt_type) & (df_res['lang'] == lang)]
            n += len(df_subset)

            if not df_subset.empty:
                df_correct_mc = df_subset[df_subset['answer'] == df_subset['prediction']]
                accuracy = (df_correct_mc.shape[0] / df_subset.shape[0])*100
                accuracies[prompt_type][lang] = round(accuracy, 2)

        # Calculate average accuracy for each prompt_type
        accuracies[prompt_type]['avg_score'] = round(sum(accuracies[prompt_type].values()) / len(accuracies[prompt_type]), 2)
    
    # Calculate average accuracy for all prompt_types
    accuracies['avg_score_all'] = round(sum([v['avg_score'] for v in accuracies.values()]) / len(accuracies), 2)
    
    accuracies['num_samples'] = n
    accuracies['model'] = model
    accuracies_mapped = final_cat_mapping(accuracies)

    with open(ACCURACY_MC_PATH.format(model=model), 'w') as f:
        json.dump(accuracies_mapped, f, indent=4)
    
    #save errors into txt file ./error/error_mc_{model}.txt to ease debugging
    with open(ERROR_MC_PATH.format(model=model), 'w') as f:
        for err in errs:
            f.write(err)
            f.write('\n')

def score_oe(model, df_res):
    oe_scores = {}
    
    prompt_types = df_res['prompt_type'].unique()
    langs = df_res['lang'].unique()
    n = 0

    # Calculate str.contain accuracy
    for prompt_type in tqdm(prompt_types, total=len(prompt_types), desc="Calculating String.find Accuracy"):
        oe_scores[prompt_type] = {}
        
        for lang in tqdm(langs, total=len(langs), desc="Processing languages for String Accuracy"):
            df_subset = df_res[(df_res['type'] == 'oe') & (df_res['prompt_type'] == prompt_type) & (df_res['lang'] == lang)].copy()  # Create a copy to avoid SettingWithCopyWarning
            n += len(df_subset)

            if not df_subset.empty:
                df_subset['answer'] = df_subset['answer'].fillna('')  # Replace NaN with empty string in 'answer'
                df_subset['prediction'] = df_subset['prediction'].fillna('')  # Replace NaN with empty string in 'prediction'

                df_subset.loc[:, 'correct'] = df_subset.apply(lambda x: x['answer'].lower() in x['prediction'].lower(), axis=1)
                accuracy = df_subset['correct'].sum() / len(df_subset) * 100
                oe_scores[prompt_type][lang] = round(accuracy, 2)
        
        # Calculate average accuracy for each prompt_type
        oe_scores[prompt_type]['avg_score'] = round(sum(oe_scores[prompt_type].values()) / len(oe_scores[prompt_type]), 2)
    
    # Calculate average accuracy for all prompt_types
    oe_scores['avg_score_all'] = round(sum([v['avg_score'] for v in oe_scores.values()]) / len(oe_scores), 2)
    
    oe_scores['num_samples'] = n
    oe_scores['model'] = model
    oe_scores_mapped = final_cat_mapping(oe_scores)
    
    with open(ACCURACY_OE_PATH.format(model=model), 'w') as f:
        json.dump(oe_scores_mapped, f, indent=4)


def score_bert_oe(model, df_res):
    bertscore = load('bertscore')
    BERTScoreModel = "xlm-roberta-large"
    bert_scores = {}
    n = 0

    prompt_types = df_res['prompt_type'].unique()
    langs = df_res['lang'].unique()

    try:
        for prompt_type in tqdm(prompt_types, total=len(prompt_types), desc="Calculating BERT scores"):
            bert_scores[prompt_type] = {}  # Initialize a sub-dictionary for each prompt_type
            
            for lang in tqdm(langs, total=len(langs), desc="Processing languages for BERT"):
                df_subset = df_res[(df_res['type'] == 'oe') & (df_res['prompt_type'] == prompt_type) & (df_res['lang'] == lang)]
                n += len(df_subset)

                if not df_subset.empty:
                    references = df_subset['answer'].astype(str).tolist()
                    predictions = df_subset['prediction'].astype(str).tolist()

                    bert_result = bertscore.compute(predictions=predictions, references=references, model_type=BERTScoreModel)
                    bert_scores[prompt_type][lang] = round(sum(bert_result['f1']) / len(bert_result['f1']) * 100, 2)

            # Calculate average BERT score for each prompt_type
            bert_scores[prompt_type]['avg_score'] = round(sum(bert_scores[prompt_type].values()) / len(bert_scores[prompt_type]), 2)

        # Calculate average BERT score for all prompt_types
        bert_scores['avg_score_all'] = round(sum([v['avg_score'] for v in bert_scores.values()]) / len(bert_scores), 2)
        
        bert_scores['num_samples'] = n
        bert_scores['model'] = model
        bert_scores_mapped = final_cat_mapping(bert_scores)
        
        with open(BERTSCORE_OE_PATH.format(model=model), 'w') as f:
            json.dump(bert_scores_mapped, f, indent=4)        

    except Exception as e:
        print(f"An error occurred during BERT calculations: {e}")


if __name__ == "__main__":
    mode = CONFIG['mode']
    models = CONFIG['models']
    
    for model in tqdm(models, total = len(models)):
        df_res, vqa_task1, vqa_task2 = load_result(model)

        if mode == "all" or mode == "mc":
            print(f"> Running [{model}] - [MC]...")
            try:
                score_mc(model, df_res, vqa_task1, vqa_task2)
            except Exception as e:
                print(f"> [{model}] - [MC]\n> An error occurred: {e}")
                continue
        
        if mode == "all" or mode == "oe":
            print(f"> Running [{model}] - [OE]...")
            try:
                score_oe(model, df_res)
                score_bert_oe(model, df_res)
            except Exception as e:
                print(f"> [{model}] - [OE]\n> An error occurred: {e}")
                continue
