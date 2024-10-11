import pandas as pd
import json
from datasets import load_dataset
import re
from tqdm import tqdm
from collections import OrderedDict
import argparse
import yaml

def load_jsonl(file_path):
  data = []
  with open(file_path, 'r') as f:
    for line in f:
      data.append(json.loads(line))
  return data

def load_result(model):
    task1_mc = pd.DataFrame(load_jsonl(f'../result/task1_mc_{model}_pred.jsonl'))
    # task1_oe = pd.DataFrame(load_jsonl(f'task1_oe_{model}_pred.jsonl'))
    task2_mc = pd.DataFrame(load_jsonl(f'../result/task2_mc_{model}_pred.jsonl'))
    # task2_oe = pd.DataFrame(load_jsonl(f'task2_oe_{model}_pred.jsonl'))

    res = [
        task1_mc,
        # task1_oe,
        task2_mc,
        # task2_oe
    ]

    ds1 = load_dataset("world-cuisines/vqa", "task1", split = "test_large")
    ds2 = load_dataset("world-cuisines/vqa", "task2", split = "test_large")
    vqa_task1 = ds1.to_pandas()
    vqa_task2 = ds2.to_pandas()

    qa_id_prompt_type_dict = vqa_task1.set_index('qa_id')['prompt_type'].to_dict()
    qa_id_prompt_type_dict.update(vqa_task2.set_index('qa_id')['prompt_type'].to_dict())

    df_res = pd.concat(res, ignore_index = True)
    df_res['prompt_type'] = df_res.qa_id.map(qa_id_prompt_type_dict)

    df_res['type'] = 'oe'
    df_res.loc[df_res.multi_choice_answer.notna(), 'type'] = 'mc'
    df_res.loc[df_res.multi_choice_answer.notna(), 'answer'] = df_res.loc[df_res.multi_choice_answer.notna(), 'multi_choice_answer']
    df_res = df_res.drop(columns=['multi_choice_answer'])

    return df_res, vqa_task1, vqa_task2

def score_mc(model):
    df_res, vqa_task1, vqa_task2 = load_result(model)

   # Regex pattern to match "<int>. " at the start of the string
    pattern = r"^(\d+)\.\s"

    for index, row in tqdm(df_res[df_res['type'] == 'mc'].iterrows(), total=len(df_res[df_res['type'] == 'mc'])):
        # Check if answer is number only
        try:
            prediction = row['prediction']
            df_res.loc[index, 'prediction'] = int(prediction)
        except ValueError:
            # Check if answer starts with number
            try:
                matched = re.match(pattern, str(prediction))
                if matched:
                    extracted_int = int(matched.group(1))
                    df_res.loc[index, 'prediction'] = extracted_int
                else:
                    raise ValueError(f"No integer found in prediction: {prediction}")
            except ValueError:
                # check if answer is one of the multiple choices, even though there are no number
                try:
                    if row['prompt_type'] == '2':
                        vqa_row = vqa_task2[(vqa_task2['qa_id'] == row['qa_id']) & (vqa_task2['lang'] == row['lang'])]
                        prompt = vqa_row['multi_choice_prompt'].iloc[0]
                        matched = prompt.find(prediction)
                        if matched != -1:
                            df_res.loc[index, 'prediction'] = prompt[matched - 3]
                        else:
                            raise ValueError(f"No answer found in prediction: {prediction}")
                    else:
                        vqa_row = vqa_task1[(vqa_task1['qa_id'] == row['qa_id']) & (vqa_task1['lang'] == row['lang']) & (vqa_task1['prompt_type'] == row['prompt_type'])]
                        prompt = vqa_row['multi_choice_prompt'].iloc[0]
                        matched = prompt.find(prediction)
                        if matched != -1:
                            df_res.loc[index, 'prediction'] = prompt[matched - 3]
                        else:
                            raise ValueError(f"No answer found in prediction: {prediction}")
                except ValueError as e:
                    print(f"Error in row {index}: {row['prediction']} - {e}\n")

    accuracies = {}
    # calculate the accuracy for each task and each language
    try:
        for prompt_type in tqdm(df_res['prompt_type'].unique(), total = len(df_res['prompt_type'].unique())):
            accuracies[prompt_type] = {}  # Initialize a sub-dictionary for each prompt_type
            for lang in tqdm(df_res['lang'].unique(), total = len(df_res['lang'].unique())):
                df_subset = df_res[(df_res['type'] == 'mc') & (df_res['prompt_type'] == prompt_type) & (df_res['lang'] == lang)]

                if not df_subset.empty:
                    df_correct_mc = df_subset[df_subset['answer'] == df_subset['prediction']]
                    accuracy = df_correct_mc.shape[0] / df_subset.shape[0]
                    accuracies[prompt_type][lang] = round(accuracy, 3)

                accuracies[prompt_type]['avg_score'] = round(sum(accuracies[prompt_type].values()) / len(accuracies[prompt_type]), 3)
    except Exception as e:
        print(f"An error occurred: {e}")

    mapkey = {
        "1": "1a. Dish Name (No-Context)",
        "2": "2. Regional Cuisine",
        "3": "1b. Dish Name (Contextualized)",
        "4": "1c. Dish Name (Adversarial)"
    }

    for category in accuracies:
        # rename language key
        if isinstance(accuracies[category], dict):
            if 'su_loma' in accuracies[category]:
                accuracies[category]['su'] = accuracies[category].pop('su_loma')
            if 'si_formal_spoken' in accuracies[category]:
                accuracies[category]['si'] = accuracies[category].pop('si_formal_spoken')
            accuracies[category] = OrderedDict(sorted(accuracies[category].items()))

        # map dictionary using new key
        accuracies_mapped = {mapkey.get(k, k): v for k, v in accuracies.items()}
        accuracies_mapped['model'] = model

    #save into json file
    with open(f'./json/accuracy_mc_{model}.json', 'w') as f:
        json.dump(accuracies_mapped, f, indent=4)

if __name__ == "__main__":
    with open('score.yml', 'r') as file:
     models = yaml.safe_load(file)

    for _, model in models.items():
        score_mc(model)




