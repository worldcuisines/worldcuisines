# 🥘 WorldCuisines: Multilingual Multicultural VQA Benchmark
![Pull Requests Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

![WorldCuisines Preview](assets/worldcuisines.png)

Introducing **WorldCuisines 🥘**, a massive-scale multilingual and multicultural VQA benchmark that challenges Vision-Language Models (VLMs) to understand cultural food diversity in over **30 languages and dialects**, across **9 language families**, with over **1 million data points** available.

### Key Stats:
- **Over 1 Million** text-image pairs
- Coverage of **2.4k** dishes with **6k** images.
- Coverage of **30 languages** across **9 language families**


## Table of Contents

- [🥘 WorldCuisines: Multilingual Multicultural VQA Benchmark](#-worldcuisines-multilingual-multicultural-vqa-benchmark)
    - [Key Stats:](#key-stats)
  - [Table of Contents](#table-of-contents)
  - [📜 Paper](#-paper)
  - [📊 Benchmark](#-benchmark)
  - [⚡ Environment Setup](#-environment-setup)
    - [Via `pip`](#via-pip)
    - [Via `conda`](#via-conda)
  - [💯 Experiment Result](#-experiment-result)
  - [🧪 Running Experiments (TODO)](#-running-experiments-todo)
  - [📈 Aggregating Experiment Result](#-aggregating-experiment-result)
  - [🏞️ Visualizing the Scores](#️-visualizing-the-scores)
    - [Examples of Radar Plot](#examples-of-radar-plot)
    - [Examples of Other Plots](#examples-of-other-plots)
  - [💻 Models Support](#-models-support)
    - [Generative VLMs:](#generative-vlms)
      - [Open-Source](#open-source)
      - [Proprietary](#proprietary)
  - [🚀 How to Contribute?](#-how-to-contribute)
  - [✏️ On Progress](#️-on-progress)

## 📜 Paper 
This is the source code of the paper [[Arxiv]](https://arxiv.org/abs/2410.12705):

This code has been written using Python. If you use any code or datasets from this toolkit in your research, please cite the associated paper.
<pre>
@misc{winata2024worldcuisinesmassivescalebenchmarkmultilingual,
      title={WorldCuisines: A Massive-Scale Benchmark for Multilingual and Multicultural Visual Question Answering on Global Cuisines}, 
      author={Genta Indra Winata and Frederikus Hudi and Patrick Amadeus Irawan and David Anugraha and Rifki Afina Putri and Yutong Wang and Adam Nohejl and Ubaidillah Ariq Prathama and Nedjma Ousidhoum and Afifa Amriani and Anar Rzayev and Anirban Das and Ashmari Pramodya and Aulia Adila and Bryan Wilie and Candy Olivia Mawalim and Ching Lam Cheng and Daud Abolade and Emmanuele Chersoni and Enrico Santus and Fariz Ikhwantri and Garry Kuwanto and Hanyang Zhao and Haryo Akbarianto Wibowo and Holy Lovenia and Jan Christian Blaise Cruz and Jan Wira Gotama Putra and Junho Myung and Lucky Susanto and Maria Angelica Riera Machin and Marina Zhukova and Michael Anugraha and Muhammad Farid Adilazuarda and Natasha Santosa and Peerat Limkonchotiwat and Raj Dabre and Rio Alexander Audino and Samuel Cahyawijaya and Shi-Xiong Zhang and Stephanie Yulia Salim and Yi Zhou and Yinxuan Gui and David Ifeoluwa Adelani and En-Shiun Annie Lee and Shogo Okada and Ayu Purwarianti and Alham Fikri Aji and Taro Watanabe and Derry Tanti Wijaya and Alice Oh and Chong-Wah Ngo},
      year={2024},
      eprint={2410.12705},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.12705}, 
}
</pre>

## 📊 Benchmark

WorldCuisines 🥘 comprises a balanced proportion of its **2 supported tasks**. We provide over **1M training data** and a **60k evaluation data**.

![WorldCuisines Dataset Statistic](assets/data_stat.png)

Our benchmark evaluates VLMs on two tasks: dish name prediction and dish location prediction. The settings include **no-context**, **contextualized**, and **adversarial** infused prompt as the model's input.

![WorldCuisines Tasks](assets/tasks.png)

## ⚡ Environment Setup

### Via `pip`
```
pip install -r requirements.txt
```
### Via `conda`
```
conda env create -f env.yml
```

## 💯 Experiment Result
If you wish to get the final result for all LMMs that we evaluate, please refer to this [leaderboard](https://huggingface.co/spaces/worldcuisines/worldcuisines) for the summary. The raw results are placed in the `evaluation/score/json` directory.

## 🧪 Running Experiments (TODO)
All experiment results will be stored in the `evaluation/result/` directory. You can execute each experiment using the following commands:

```
cd evaluation/
python {model_name}.py --model_path {model_path} --task {task} --type {type} 
```
**Main Arguments:**
- `model_name`: choose from `aria`, `gemini`, `llama`, `llava`, `molmo`, `phi`, `qwen`, or `pixtral`
- `model_path`: Hugging Face model handle (e.g., `rhymes-ai/Aria`)
- `task`: `1` or `2`, refer to [this](#-benchmark) for details
- `type`: `mc` (multiple-choice) or `oe` (open-ended)

**Other Arguments:**
- **TODO**

## 📈 Aggregating Experiment Result 
Edit `evaluation/score/score.yml` to determine scoring mode, evaluation set, and evaluated VLMs. Note that `mc` means multiple-choice and `oe` means open-ended.

```yml
mode: all # {all, mc, oe}  all = mc + oe
oe_mode: multi # {single, dual, multi}
subset: large # {large, small}
models:
- llava-1.6-7b
- llava-1.6-13b
- qwen-vl-2b
- qwen2-vl-7b-instruct
- qwen2-vl-72b
- llama-3.2-11b
- llama-3.2-90b
- molmoe-1b
- molmo-7b-d
- molmo-7b-o
- aria-25B-moe-4B
- Phi-3.5-vision-instruct
- pixtral-12b
- nvlm
- gpt-4o-2024-08-06
- gpt-4o-mini-2024-07-18
- gemini-1.5-flash
```

In addition to the `multi` mode for generating the `oe` score, which compares the answer to the golden labels across all languages, we also support other golden label referencing settings:

- **`single` reference**: compares the answer only to the golden label in the original language.
- **`dual` reference**: compares the answer to the golden label in the original language and English.

Once set, run this command:
```bash
cd evaluation/score/
python score.py
```


## 🏞️ Visualizing the Scores

We provide radar, scatter, and connected scatter-line plots to visualize scoring results for all VLMs in `evaluation/score/plot/`.

To generate all **radar plot**, use:
```
python evaluation/score/plot/visualization.py
```

### Examples of Radar Plot
![Radar Plot Example](evaluation/score/plot/radar_avg_mc_combined.png)

You can also modify `evaluation/score/score.yml` to select which VLMs to visualize and adjust plot labels in `plot_mapper.yml`.

### Examples of Other Plots

<img src="assets/model_params.png" width="30%"> <img src="assets/model_scatter.png" width="33%"> <img src="assets/scatterplot.png" width="25%">


Other plot generation scripts are available in the `*.ipynb` files within the same directory.

## 💻 Models Support
Our codebase supports the usage of multiple models for the experiments, providing flexibility for customization of the list shown below:

### Generative VLMs:
#### Open-Source
- Llava1.6 Vicuna [llava-hf/llava-v1.6-vicuna-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf) [llava-hf/llava-v1.6-vicuna-13b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf)
- Qwen2 VL Instruct [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) [Qwen/Qwen2-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct)
- Llama 3.2 Instruct [meta-llama/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) [meta-llama/Llama-3.2-90B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct)
- Molmo-E 1B [allenai/MolmoE-1B-0924](https://huggingface.co/allenai/MolmoE-1B-0924)
- Molmo-D 7B [allenai/Molmo-7B-D-0924](https://huggingface.co/allenai/Molmo-7B-D-0924)
- Molmo-O 7B [allenai/Molmo-7B-O-0924](https://huggingface.co/allenai/Molmo-7B-O-0924)
- Aria 25B  [rhymes-ai/Aria](https://huggingface.co/rhymes-ai/Aria)
- Phi-3.5 Vision 4B [microsoft/Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)
- Pixtral 12B [mistralai/Pixtral-12B-2409](https://huggingface.co/mistralai/Pixtral-12B-2409)

#### Proprietary 
(last tested as of October 2024)
- GPT-4o
- GPT-4o Mini
- Gemini 1.5 Flash

## 🚀 How to Contribute?
Feel free to create [an issue](https://github.com/worldcuisines/worldcuisines/issues) if you have any questions. And, create [a PR](https://github.com/worldcuisines/worldcuisines/pulls) for fixing bugs or adding improvements.

If you are interested to create an extension of this work, feel free to reach out to [us](mailto:gentaindrawinata@gmail.com)!

Support our open source effort ⭐

## ✏️ On Progress
We are improving the code, especially on inference part to generate `evaluation/result` and scoring visualization, to make it more user-friendly and customizable.
