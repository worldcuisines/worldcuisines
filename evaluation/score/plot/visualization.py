import json
import numpy as np
import matplotlib.pyplot as plt
import yaml
from collections import OrderedDict, defaultdict
import numpy as np

with open('./plot_mapper.yml', 'r') as file:
    plot_mapper = yaml.safe_load(file)

LANG_FAMILY_DICT = plot_mapper['lang_family_dict']
LANG_RESOURCE_DICT = plot_mapper['lang_resource_dict']
CATEGORIES = plot_mapper['categories']
MODEL_NAME_DICT = plot_mapper['model_name_dict']
TASK_DICT = plot_mapper['task_dict']

ACCURACY_MC_PATH = "../json/{model}_accuracy_mc.json"
ACCURACY_OE_PATH = "../json/{model}_accuracy_oe_multi.json"
BERTSCORE_OE_PATH = "../json/{model}_bertscore_oe.json"


def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def generate_accuracy_dict(task, vis_type):
    with open('../score.yml', 'r') as file:
        models = yaml.safe_load(file)
    models = models['models']

    model_dict = {}
    if vis_type == 'lang':
        if task == 'mc':
            model_dict = {
                model: load_json(ACCURACY_MC_PATH.format(model=model)) for model in models
            }
        elif task == 'oe':
            model_dict = {
                model: load_json(ACCURACY_OE_PATH.format(model=model)) for model in models
            }
        else:
            model_dict = {
                model: load_json(BERTSCORE_OE_PATH.format(model=model)) for model in models
            }
    else:
        for model in models:
            if task == 'mc':
                model_result = load_json(ACCURACY_MC_PATH.format(model=model))
            elif task == 'oe':
                model_result = load_json(ACCURACY_OE_PATH.format(model=model))
            else:
                model_result = load_json(BERTSCORE_OE_PATH.format(model=model))
            for category in CATEGORIES:
                category_totals = defaultdict(list)
                for lang_code, score in model_result[category].items():
                    if vis_type == 'res':
                        conversion_dict = LANG_RESOURCE_DICT
                    else:
                        conversion_dict = LANG_FAMILY_DICT
                    if lang_code in conversion_dict:
                        category_totals[conversion_dict[lang_code]].append(score)
                category_averages = {category: sum(scores) / len(scores) for category, scores in category_totals.items()}
                category_averages = OrderedDict(sorted(category_averages.items()))
                model_result[category] = category_averages
            model_dict[model] = model_result

    return model_dict

def average_score_dict(task, vis_type):
    data = generate_accuracy_dict(task, vis_type)
    avg_data = {}
    model_keys = list(data.keys())

    metrics = list(data[model_keys[0]].keys())

    for metric in metrics:
        if isinstance(data[model_keys[0]][metric], dict):
            avg_data[metric] = {}
            for dish in data[model_keys[0]][metric]:
                avg_data[metric][dish] = round(np.mean([data[model][metric][dish] for model in model_keys]),2)
        elif isinstance(data[model_keys[0]][metric], float) or isinstance(data[model_keys[0]][metric], int):
            avg_data[metric] = round(np.mean([data[model][metric] for model in model_keys]),2)

    avg_data["model"] = "avg_all_model_score"

    return avg_data

def plot_radar(models, data, title, ax, task, vis_type):
    if vis_type == 'lang':
        _languages = [
            # 'bn', 'cs', 'en', 'es', 'fr', 'hi', 'it', 'mr', 'ru_casual', 'ru_formal', 'sc', 'si',  # Indo-European
            'en', 'es', 'fr',
            'cs', 'it', 'ru_casual', 'ru_formal',
            'bn', 'hi', 'mr', 'sc', 'si',  # Indo-European
            'id_casual', 'id_formal', 'tl', 'jv_krama', 'jv_ngoko', 'su',  # Austronesian
            'nan', 'nan_spoken', 'yue', 'zh_cn',  # Sino-Tibetan
            'ja_casual', 'ja_formal',  # Japonic
            'ko_casual', 'ko_formal',  # Koreanic
            'th',  # Kra-Dai
            'yo',  # Niger-Congo
            'ar',  # Afro-Asiatic
            'az',  # Turkic
        ]
        languages = [key for key in _languages if key in data[models[0]].keys()]
    else:
        languages = [key for key in data[models[0]].keys() if key != 'avg_score']
    N = len(languages)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    handles = []
    labels = []

    hex_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080']

    color_id = 0
    for model in models:
        values = [data[model][lang] for lang in languages]
        values += values[:1]
        handle, = ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=hex_colors[color_id])
        ax.fill(angles, values, alpha=0.05)
        handles.append(handle)
        labels.append(model)
        color_id += 1

    if vis_type == 'lang':
        ax.set_title(TASK_DICT[title], size=12, pad=40)
    else:
        ax.set_title(TASK_DICT[title], size=12, pad=30)

    if task == "mc":
        ax.set_ylim(max(int(np.min(values) / 20), 0) * 10, 100)
        ax.set_yticks(range(max(int(np.min(values) / 20), 0) * 20, 100, 20))
    elif task == "oe":
        ax.set_ylim(max(int(np.min(values) / 20), 0) * 10, 50)
        ax.set_yticks(range(max(int(np.min(values) / 20), 0) * 20, 50, 20))
    else:
        ax.set_ylim(max(int(np.min(values) / 20), 0) * 10, 100)
        ax.set_yticks(range(max(int(np.min(values) / 20), 0) * 20, 100, 20))

    ticks = angles[:-1]
    ax.set_xticks(ticks)
    ax.set_xticklabels([''] * len(languages))

    base_rotation = 90  # First label starts at -90 degrees
    rotation_increment = -12  # Each subsequent label is rotated 12 degrees more

    if vis_type == 'lang':
        # Manually set each xticklabel with incremented rotation
        for i, (tick, label) in enumerate(zip(ticks, languages)):
            rotation = base_rotation + i * rotation_increment
            if rotation <= -90:
                rotation += 180
            ax.text(tick, ax.get_rmax() * (1.06 + (len(label) - 2) * 0.035), label, rotation=rotation, size=10, ha='center', va='center')
    elif vis_type == 'res':
        for i, (tick, label) in enumerate(zip(ticks, languages)):
            if i == 0 or i == 3:
                ax.text(tick, ax.get_rmax() * 1.05, label, size=9, ha='center', va='center')
            if i == 1 or i == 2:
                ax.text(tick, ax.get_rmax() * 1.05, label, size=9, ha='left', va='center')
            if i == 4 or i == 5:
                ax.text(tick, ax.get_rmax() * 1.05, label, size=9, ha='right', va='center')
    else:
        for i, (tick, label) in enumerate(zip(ticks, languages)):
            if i == 0:
                ax.text(tick, ax.get_rmax() * 1.05, label, size=9, ha='center', va='center')
            if i in [1, 2, 3, 4]:
                ax.text(tick, ax.get_rmax() * 1.05, label, size=9, ha='left', va='center')
            if i in [5, 6, 7, 8]:
                ax.text(tick, ax.get_rmax() * 1.05, label, size=9, ha='right', va='center')

    return handles, labels


def save_radar(task, vis_type):
    with open('../score.yml', 'r') as file:
        models = yaml.safe_load(file)
    models = models['models']
    model_dict = generate_accuracy_dict(task, vis_type)

    # Create subplots for the 4 radar charts
    if vis_type == 'lang':
        fig, axs = plt.subplots(2, 2, figsize=(10, 10), subplot_kw=dict(polar=True))
    else:
        fig, axs = plt.subplots(2, 2, figsize=(10, 10), subplot_kw=dict(polar=True))

    # Initialize lists to collect handles and labels for the suplegend
    handles_list = []
    labels_list = []

    # Plot radar charts for each category
    for i, category in enumerate(CATEGORIES):
        row, col = divmod(i, 2)
        category_data = {model: model_dict[model][category] for model in models}
        handles, labels = plot_radar(models, category_data, category, axs[row, col], task, vis_type)
        handles_list = handles
        labels_list = labels

    # if task == 'mc':
    #     fig.suptitle('Multiple Choice Accuracy', fontsize=16)
    # elif task == 'oe'
    #     fig.suptitle('Open Ended Accuracy', fontsize=16)
    # else:
    #     fig.suptitle('Open Ended BERT Score', fontsize=16)

    fig.legend(handles_list, list(map(MODEL_NAME_DICT.get, labels_list)), loc='lower center', ncol=len(models) // 4)
    if vis_type == 'lang':
        plt.subplots_adjust(bottom=0.2, wspace = 0.7, hspace=1.5)
        plt.tight_layout(rect=[0, 0.1, 1, 0.98])
    else:
        plt.subplots_adjust(bottom=0.8, wspace = 20)
        plt.tight_layout(rect=[0, 0.1, 1, 0.98])
    # plt.tight_layout()
    # plt.subplots_adjust(bottom=0.07)
    plt.savefig(f'radar_{task}_{vis_type}')



def plot_single_model_radar(data, ax, mc_oe, vis_type):
    with open('./plot_mapper.yml', 'r') as file:
        tasks = yaml.safe_load(file)['categories']
    languages = list(data[tasks[0]].keys())
    if "avg_score" in languages:
        languages.remove("avg_score")
    N = len(languages)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    hex_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe']

    handles = []
    labels = []

    for i, task in enumerate(tasks):
        values = [data[task].get(lang, 0) for lang in languages]  # Default to 0 if language is missing
        values += values[:1]  # Close the radar loop

        handle, = ax.plot(angles, values, linewidth=2, linestyle='solid', label=task, color=hex_colors[i % len(hex_colors)])
        ax.fill(angles, values, alpha=0.1, color=hex_colors[i % len(hex_colors)])
        handles.append(handle)
        labels.append(task)

    min_limit = 0 
    max_limit = 80
    min_label = 0
    max_label = 61
    jump = 20

    if mc_oe == 'oe':
        max_limit = 30
        max_label = 21
        jump = 10
    elif mc_oe == 'bs':
        min_limit = 40
        max_limit = 100
        min_label = 41
        max_label = 101
    print(min_limit, max_limit, min_label, max_label, jump)

    ax.set_ylim(min_limit, max_limit)
    ax.set_yticks(range(min_label, max_label, jump))

    # Radar chart tick and label arrangement
    ticks = angles[:-1]
    ax.set_xticks(ticks)
    ax.set_xticklabels([''] * len(languages))  # Hide the default xtick labels

    base_rotation = 90  # First label starts at 90 degrees
    rotation_increment = -12  # Each subsequent label is rotated 12 degrees more

    if vis_type == 'lang':
        # Manually set each xticklabel with incremented rotation
        for i, (tick, label) in enumerate(zip(ticks, languages)):
            rotation = base_rotation + i * rotation_increment
            if rotation <= -90:
                rotation += 180
            ax.text(tick, ax.get_rmax() * (1.1 + (len(label) if len(label) <= 3 else len(label) - 1) * 0.03), label, rotation=rotation, size=14, ha='center', va='center')
    elif vis_type == 'res':
        for i, (tick, label) in enumerate(zip(ticks, languages)):
            if i == 0 or i == 3:
                ax.text(tick, ax.get_rmax() * 1.1 +0.3, label, size=14, ha='center', va='center')
            elif i == 1 or i == 2:
                ax.text(tick, ax.get_rmax() * 1.1, label, size=14, ha='left', va='center')
            elif i == 4 or i == 5:
                ax.text(tick, ax.get_rmax() * 1.1, label, size=14, ha='right', va='center')
    else:
        for i, (tick, label) in enumerate(zip(ticks, languages)):
            if i == 0:
                ax.text(tick, ax.get_rmax() * 1.1 + 0.3, label, size=14, ha='center', va='center')
            elif i in [1, 2, 3, 4]:
                ax.text(tick, ax.get_rmax() * 1.1, label, size=14, ha='left', va='center')
            elif i in [5, 6, 7, 8]:
                ax.text(tick, ax.get_rmax() * 1.1, label, size=14, ha='right', va='center')

    return handles, labels

# Usage
def save_single_model_radar(task ,vis_type):
    fig, ax = plt.subplots(figsize=(6, 7), subplot_kw=dict(polar=True))
    data = average_score_dict(task, vis_type)
    
    handles, labels = plot_single_model_radar(data, ax, task, vis_type)
    
    fig.legend(handles, labels, loc='lower center', ncol=2,)
    plt.tight_layout()
    plt.savefig(f'radar_avg_combined_{task}_{vis_type}.png', dpi=300)


def save_single_model_radar_combined(task, vis_types):
    fig, axs = plt.subplots(1, 3, figsize=(18, 7), subplot_kw=dict(polar=True))

    all_handles, all_labels = [], []

    for ax, vis_type in zip(axs, vis_types):
        data = average_score_dict(task, vis_type)
        handles, labels = plot_single_model_radar(data, ax, task, vis_type)
        all_handles = handles  # Since handles and labels are the same for all plots
        all_labels = labels

    # Combine legends for all plots
    fig.legend(all_handles, all_labels, loc='lower center', ncol=4, fontsize=16, bbox_to_anchor=(0.5, 0.06))

    # Adjust layout so legend doesn't overlap the plots
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(f'radar_avg_{task}_combined.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    # save_radar('mc', 'lang')
    # save_radar('mc', 'res')
    # save_radar('mc', 'fam')

    # save_radar('oe', 'lang')
    # save_radar('oe', 'res')
    # save_radar('oe', 'fam')

    # save_radar('bs', 'lang')
    # save_radar('bs', 'res')
    # save_radar('bs', 'fam')

    # save_single_model_radar('mc', 'lang')
    # save_single_model_radar('mc', 'res')
    # save_single_model_radar('mc', 'fam')

    # save_single_model_radar('oe', 'lang')
    # save_single_model_radar('oe', 'res')
    # save_single_model_radar('oe', 'fam')

    save_single_model_radar_combined('mc', ['lang', 'res', 'fam'])
    save_single_model_radar_combined('oe', ['lang', 'res', 'fam'])
    save_single_model_radar_combined('bs', ['lang', 'res', 'fam'])
