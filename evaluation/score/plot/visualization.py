import json
import numpy as np
import matplotlib.pyplot as plt
import yaml
from collections import OrderedDict, defaultdict

with open('./plot_mapper.yml', 'r') as file:
    plot_mapper = yaml.safe_load(file)

LANG_FAMILY_DICT = plot_mapper['lang_family_dict']
LANG_RESOURCE_DICT = plot_mapper['lang_resource_dict']
CATEGORIES = plot_mapper['categories']
MODEL_NAME_DICT = plot_mapper['model_name_dict']

ACCURACY_MC_PATH = "../json/{model}_accuracy_mc.json"
ACCURACY_OE_PATH = "../json/{model}_accuracy_oe.json"
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


def plot_radar(models, data, title, ax, task, vis_type):
    languages = [key for key in data[models[0]].keys() if key != 'avg_score']
    N = len(languages)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    handles = []
    labels = []
    for model in models:
        values = [data[model][lang] for lang in languages]
        values += values[:1]
        handle, = ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=0.05)
        handles.append(handle)
        labels.append(model)

    if vis_type == 'lang':
        ax.set_title(title, size=12, pad=40)
    else:
        ax.set_title(title, size=12, pad=30)

    if task == 'mc':
        ax.set_ylim(0, 100)
        ax.set_yticks([10, 20, 30, 40, 50, 60, 70, 80, 90])
        ax.set_yticklabels(['10', '20', '30', '40', '50', '60', '70', '80', '90'])
    elif task == 'oe':
        ax.set_ylim(0, 50)
        ax.set_yticks([5, 10, 15, 20, 25, 30, 35, 40, 45]) 
        ax.set_yticklabels(['5', '10', '15', '20', '25', '30', '35', '40', '45']) 
    else:
        ax.set_ylim(0, 100)
        ax.set_yticks([10, 20, 30, 40, 50, 60, 70, 80, 90])
        ax.set_yticklabels(['10', '20', '30', '40', '50', '60', '70', '80', '90'])

    ticks = angles[:-1]
    ax.set_xticks(ticks)
    ax.set_xticklabels([''] * len(languages))

    base_rotation = 90  # First label starts at -90 degrees
    rotation_increment = -12  # Each subsequent label is rotated 12 degrees more

    if vis_type == 'lang':
        # Manually set each xticklabel with incremented rotation
        for i, (tick, label) in enumerate(zip(ticks, languages)):
            rotation = base_rotation + i * rotation_increment
            if rotation < -90:
                rotation += 180
            ax.text(tick, ax.get_rmax()*(1.06 + (len(label)-2)*0.02), label, rotation=rotation, size=10, ha='center', va='center')
    elif vis_type == 'res':
        for i, (tick, label) in enumerate(zip(ticks, languages)):
            if i == 0 or i==3:
                ax.text(tick, ax.get_rmax()*1.05, label, size=10, ha='center', va='center')
            if i == 1 or i==2:
                ax.text(tick, ax.get_rmax()*1.05, label, size=10, ha='left', va='center')
            if i == 4 or i==5:
                ax.text(tick, ax.get_rmax()*1.05, label, size=10, ha='right', va='center')
    else:
        for i, (tick, label) in enumerate(zip(ticks, languages)):
            if i == 0:
                ax.text(tick, ax.get_rmax()*1.05, label, size=10, ha='center', va='center')
            if i in [1,2,3,4]:
                ax.text(tick, ax.get_rmax()*1.05, label, size=10, ha='left', va='center')
            if i in [5,6,7,8]:
                ax.text(tick, ax.get_rmax()*1.05, label, size=10, ha='right', va='center')

    return handles, labels


def save_radar(task, vis_type):
    with open('../score.yml', 'r') as file:
        models = yaml.safe_load(file)
    models = models['models']
    model_dict = generate_accuracy_dict(task, vis_type)

    # Create subplots for the 4 radar charts
    if vis_type == 'lang':
        fig, axs = plt.subplots(2, 2, figsize=(12, 14), subplot_kw=dict(polar=True))
    else:
        fig, axs = plt.subplots(2, 2, figsize=(14, 12), subplot_kw=dict(polar=True))

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

    if task == 'mc':
        fig.suptitle('Multiple Choice Accuracy', fontsize=16)
    elif task == 'oe':
        fig.suptitle('Open Ended Accuracy', fontsize=16)
    else:
        fig.suptitle('Open Ended BERT Score', fontsize=16)

    fig.legend(handles_list, list(map(MODEL_NAME_DICT.get, labels_list)), loc='lower center', ncol=len(models)//2)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.subplots_adjust(bottom=0.07)
    plt.savefig(f'radar_{task}_{vis_type}')


if __name__ == "__main__":
    save_radar('mc', 'lang')
    save_radar('mc', 'res')
    save_radar('mc', 'fam')

    save_radar('oe', 'lang')
    save_radar('oe', 'res')
    save_radar('oe', 'fam')

    save_radar('bs', 'lang')
    save_radar('bs', 'res')
    save_radar('bs', 'fam')
