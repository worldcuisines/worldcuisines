# qa_id,food_id,prompt_id,prompt_type,answer,answer_it,answer_sc,correct_index,image_index,image_url,Coarse-grained categories,Fine-grained categories,Cuisines,Associated Cuisines,Area,Countries,Regions,Text Description,prompt_en,prompt_id_formal,prompt_id_casual,prompt_zh-CN,prompt_ko_formal,prompt_ko_casual,prompt_ja_formal,prompt_ja_casual,prompt_su_loma,prompt_jv_krama,prompt_jv_ngoko,prompt_cs,prompt_es,prompt_fr,prompt_ar,prompt_hi,prompt_bn,prompt_mr,prompt_si_formal_spoken,prompt_yo,prompt_yue,prompt_nan,prompt_nan_spoken,prompt_tl,prompt_th,prompt_az,prompt_ru_formal,prompt_ru_casual,prompt_it,prompt_sc,answer_en,answer_id_formal,answer_id_casual,answer_zh-CN,answer_ko_formal,answer_ko_casual,answer_ja_formal,answer_ja_casual,answer_su_loma,answer_jv_krama,answer_jv_ngoko,answer_cs,answer_es,answer_fr,answer_ar,answer_hi,answer_bn,answer_mr,answer_si_formal_spoken,answer_yo,answer_yue,answer_nan,answer_nan_spoken,answer_tl,answer_th,answer_az,answer_ru_formal,answer_ru_casual,lang_status

import csv
import json
import ast
import os

### Templates ###
def generate_multiple_choices_sample(context, options, answer):
    chars = ["1", "2", "3", "4", "5"]
    answer_prompt = ""
    gold_label = 0

    for opt_id in range(len(options)):
        opt = options[opt_id]
        answer_prompt += f"{chars[opt_id]}. {opt}\n"

        if options[opt_id] == answer:
            gold_label = opt_id + 1

    user_prompt = f"{context}\n{answer_prompt}\nPrint only the answer with a single answer id (1,2,3,4,5)."
    return user_prompt, gold_label

def generate_open_ended_sample(context, options, answer):
    answer_prompt = ""
    gold_label = answer

    user_prompt = f"{context}\n{answer_prompt}\nPrint only the answer."
    return user_prompt, gold_label

FILE_NAMES = ["small_eval_task1.csv", "small_eval_task2.csv", "large_eval_task1.csv", "large_eval_task2.csv"]

langs = ["en","id_formal","id_casual","zh_cn","ko_formal","ko_casual","ja_formal","ja_casual","su_loma","jv_krama","jv_ngoko","cs","es","fr","ar","hi","bn","mr","si_formal_spoken","yo","yue","nan","nan_spoken","tl","th","az","ru_formal","ru_casual","it","sc"]

for file_name in FILE_NAMES:
    print(f"> {file_name}")
    with open(f"./generated_data/{file_name}") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        count = 0
        multi_choice_data = {}
        open_ended_data = {}
        for row in reader:
            count += 1
            if count == 1:
                continue

            qa_id,food_id,prompt_id,prompt_type,answer,answer_it,answer_sc,correct_index,image_index,image_url,coarse_categories,fine_categories,cuisines,associated_cuisines,area,countries,regions,text_descriptions,prompt_en,prompt_id_formal,prompt_id_casual,prompt_zh_cn,prompt_ko_formal,prompt_ko_casual,prompt_ja_formal,prompt_ja_casual,prompt_su_loma,prompt_jv_krama,prompt_jv_ngoko,prompt_cs,prompt_es,prompt_fr,prompt_ar,prompt_hi,prompt_bn,prompt_mr,prompt_si_formal_spoken,prompt_yo,prompt_yue,prompt_nan,prompt_nan_spoken,prompt_tl,prompt_th,prompt_az,prompt_ru_formal,prompt_ru_casual,prompt_it,prompt_sc,answer_en,answer_id_formal,answer_id_casual,answer_zh_cn,answer_ko_formal,answer_ko_casual,answer_ja_formal,answer_ja_casual,answer_su_loma,answer_jv_krama,answer_jv_ngoko,answer_cs,answer_es,answer_fr,answer_ar,answer_hi,answer_bn,answer_mr,answer_si_formal_spoken,answer_yo,answer_yue,answer_nan,answer_nan_spoken,answer_tl,answer_th,answer_az,answer_ru_formal,answer_ru_casual,lang_status = row

            prompts = [prompt_en,prompt_id_formal,prompt_id_casual,prompt_zh_cn,prompt_ko_formal,prompt_ko_casual,prompt_ja_formal,prompt_ja_casual,prompt_su_loma,prompt_jv_krama,prompt_jv_ngoko,prompt_cs,prompt_es,prompt_fr,prompt_ar,prompt_hi,prompt_bn,prompt_mr,prompt_si_formal_spoken,prompt_yo,prompt_yue,prompt_nan,prompt_nan_spoken,prompt_tl,prompt_th,prompt_az,prompt_ru_formal,prompt_ru_casual,prompt_it,prompt_sc]

            answers = [answer_en,answer_id_formal,answer_id_casual,answer_zh_cn,answer_ko_formal,answer_ko_casual,answer_ja_formal,answer_ja_casual,answer_su_loma,answer_jv_krama,answer_jv_ngoko,answer_cs,answer_es,answer_fr,answer_ar,answer_hi,answer_bn,answer_mr,answer_si_formal_spoken,answer_yo,answer_yue,answer_nan,answer_nan_spoken,answer_tl,answer_th,answer_az,answer_ru_formal,answer_ru_casual,answer_it,answer_sc]

            for i in range(len(answers)):
                answers[i] = ast.literal_eval(answers[i])

            system_prompt_map = {}
            for i in range(len(langs)):
                lang = langs[i]

                if lang not in multi_choice_data:
                    multi_choice_data[lang] = []
                    open_ended_data[lang] = []
            
                user_prompt, gold_label = generate_multiple_choices_sample(prompts[i], answers[i], answer)
                obj = {"qa_id": qa_id, "text_prompt": user_prompt, "image_url": image_url, "answer": answer, "gold_label": gold_label, "type": "multi_choice"}
                multi_choice_data[lang].append(obj)

                user_prompt, gold_label = generate_open_ended_sample(prompts[i], answers[i], answer)
                open_ended_obj = {"qa_id": qa_id, "text_prompt": user_prompt, "image_url": image_url, "answer": answer, "gold_label": gold_label, "type": "open_ended"}
                open_ended_data[lang].append(open_ended_obj)

    json_file_name = file_name.replace(".csv", ".jsonl")
    only_file_name = file_name.replace(".csv","")
    os.system(f"mkdir -p ./generated_data/prompt_multi_choice/{only_file_name}/")
    os.system(f"mkdir -p ./generated_data/prompt_open_ended/{only_file_name}/")
    for lang in langs:
        print(f"> {lang}")
        with open(f"./generated_data/prompt_multi_choice/{only_file_name}/{lang}_{json_file_name}", "w+") as out_file:
            for i in range(len(multi_choice_data[lang])):
                out_file.write(json.dumps(multi_choice_data[lang][i], ensure_ascii=False) + "\n")

        with open(f"./generated_data/prompt_open_ended/{only_file_name}/{lang}_{json_file_name}", "w+") as out_file:
            for i in range(len(open_ended_data[lang])):
                out_file.write(json.dumps(open_ended_data[lang][i], ensure_ascii=False) + "\n")