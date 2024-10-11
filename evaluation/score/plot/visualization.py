import json
import numpy as np
import matplotlib.pyplot as plt
import yaml
from collections import OrderedDict, defaultdict
from lang_dict import lang_family_dict, lang_resource_dict, categories

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def generate_accuracy_dict(task, vis_type):
    with open('../score.yml', 'r') as file:
        models = yaml.safe_load(file)
    
    model_dict = {}
    if vis_type == 'lang':
        if task == 'mc':
            model_dict = {
                model: load_json(f'../json/accuracy_mc_{model}.json') for _, model in models.items()
            }
        else:
            model_dict = {
                model: load_json(f'../json/accuracy_oe_{model}.json') for _, model in models.items()
            }
    else:
        for _, model in models.items():
            if task == 'mc':
                model_result = load_json(f'../json/accuracy_mc_{model}.json')
            else:
                model_result = load_json(f'../json/accuracy_oe_{model}.json')
            for category in categories:
                category_totals = defaultdict(list)
                for lang_code, score in model_result[category].items():
                    if vis_type == 'res':
                        conversion_dict = lang_resource_dict
                    else:
                        conversion_dict = lang_family_dict
                    if lang_code in conversion_dict:
                        category_totals[conversion_dict[lang_code]].append(score)
                category_averages = {category: sum(scores) / len(scores) for category, scores in category_totals.items()}
                category_averages = OrderedDict(sorted(category_averages.items()))
                model_result[category] = category_averages
            model_dict[model] = model_result
    
    return model_dict

def plot_radar(models, data, title, ax, vis_type):
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
        ax.fill(angles, values, alpha=0.1)
        handles.append(handle)
        labels.append(model)

    ax.set_ylim(0, 0.7)
    if vis_type == 'lang':
        ax.set_title(title, size=12, pad=40)
    else:
        ax.set_title(title, size=12, pad=30)

    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # Custom radial ticks
    ax.set_yticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6'])  # Custom tick labels

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
            ax.text(tick, ax.get_rmax() + 0.06 + (len(label)-2)*0.015, label, rotation=rotation, size=10, ha='center', va='center')
    elif vis_type == 'res':
        for i, (tick, label) in enumerate(zip(ticks, languages)):
            if i == 0 or i==3:
                ax.text(tick, ax.get_rmax() + 0.05, label, size=10, ha='center', va='center')
            if i == 1 or i==2:
                ax.text(tick, ax.get_rmax() + 0.05, label, size=10, ha='left', va='center')
            if i == 4 or i==5:
                ax.text(tick, ax.get_rmax() + 0.05, label, size=10, ha='right', va='center')
    else:
        for i, (tick, label) in enumerate(zip(ticks, languages)):
            if i == 0:
                ax.text(tick, ax.get_rmax() + 0.05, label, size=10, ha='center', va='center')
            if i in [1,2,3,4]:
                ax.text(tick, ax.get_rmax() + 0.05, label, size=10, ha='left', va='center')
            if i in [5,6,7,8]:
                ax.text(tick, ax.get_rmax() + 0.05, label, size=10, ha='right', va='center')

    return handles, labels

def save_radar(task, vis_type):
    with open('../score.yml', 'r') as file:
        models = yaml.safe_load(file)
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
    for i, category in enumerate(categories):
        row, col = divmod(i, 2)
        models = list(model_dict.keys())
        category_data = {model: model_dict[model][category] for model in models}
        handles, labels = plot_radar(models, category_data, category, axs[row, col], vis_type)
        handles_list = handles
        labels_list = labels

    if task == 'mc':
        fig.suptitle('Multiple Choice Accuracy', fontsize=16)
    else:
        fig.suptitle('Open Ended Accuracy', fontsize=16)

    fig.legend(handles_list, labels_list, loc='lower center', ncol=len(models))
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.subplots_adjust(bottom=0.06)
    plt.savefig(f'radar_{task}_{vis_type}')

if __name__ == "__main__":
    save_radar('mc', 'lang')
    save_radar('mc', 'res')
    save_radar('mc', 'fam')

    #TODO: ADD RADAR FOR OPEN ENDED