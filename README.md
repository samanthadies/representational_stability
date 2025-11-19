# Representational Stability of Truth in Large Language Models

This repository contains the full pipeline used to measure
**representational stability** in large language models (LLMs) under
controlled truth-label perturbations.\
It includes scripts for:

-   Extracting **layerwise activations** from Hugging Face models
-   Generating **noise activations**
-   Training **linear probes** (sAwMIL and Mean Difference) for "True
    vs. Not-True" classification
-   Running multi-task perturbation experiments
-   Generating all plots used in the paper (activation heatmaps,
    decision-boundary heatmaps, n-gram distributions, stability bar
    charts)

If you use this code, please include a citation to

    @article{dies2025representationalstability,
      title={Representational Stability in Large Language Models},
      author={Samantha Dies and Maynard Courtney and Germans Savcisens and Tina Eliassi-Rad},
      journal={},
      doi={},
      year={2025},
    }

The datasets with the train, test, and calibration splits can also be 
downloaded from HuggingFace. 

True, False, and Synthetic Data:

    @misc{trilemma2025data,
      title={trilemma-of-truth (Revision cd49e0e)},
      author={Germans Savcisens and Tina Eliassi-Rad},
      url={https://huggingface.co/datasets/carlomarxx/trilemma-of-truth},
      doi={10.57967/hf/5900},
      publisher={HuggingFace}
      year={2025},
    }

Fictional Data:

    @misc{stability2025data,
      title={representational-stability},
      author={Samantha Dies and Maynard Courtney and Germans Savcisens and Tina Eliassi-Rad},
      url={},
      doi={},
      publisher={HuggingFace}
      year={2025},
    }

------------------------------------------------------------------------

## Installation and Set-up

Clone the repository:
```bash
git clone https://github.com/samanthadies/representational_stability.git
cd representational_stability
```

### Environment Setup

You can recreate the exact environment using:

``` bash
micromamba env create -f environment.yml
# or
conda env create -f environment.yml
```

To activate the environment on your own system:

``` bash
conda activate stability
```

### HuggingFace Access Tokens

To get HuggingFace **Access Tokens** for gated models, visit 
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
You will need to update the ```configs/model``` files with the token
or pass it into the ```token``` field through the command line to generate
activations with the gated models.

------------------------------------------------------------------------

## Usage and Examples

We use ```Hydra``` to run and manage our experiments. Refer to the
[Hydra documentation](https://hydra.cc/docs/intro/) for help.

### 1. Collect Hidden Activations

The LLM activations are ***not*** included in this repository due to their size.
You can generate the activations for each LLM (e.g., ```llama-3-8b```)
by running

``` bash
python collect_activations.py \
  --config-path configs \
  --config-name activations \
  hydra.run.dir=. \
  model='LLM' \
  model.token='YOUR_HF_TOKEN_HERE' \
  output_dir='outputs/activations/${model.name}'
```

This stores the generated activations for all datasets and saves them at,
for example, [outputs/activations/](outputs/activations/). To generate
activations for a subset of the datasets, include the ```datasets=[]```
hydra override.

### 2. Generate Noise Activations

For each datapack (e.g., ```cities_loc```) and LLM (e.g., ```llama-3-8b```), 
generate the noise activations by running

``` bash
python collect_noise_activations.py \
  --config-path configs \
  --config-name generate_noise \
  hydra.run.dir=. \
  model='LLM' \
  datapack='DATAPACK'
```
This stores noise activations in the same directory as the real activations
and generates dummy noise datasets. To change the number of noise statements 
generated, use the hydra override ```pct_of_train_tag```. This generates noise 
as a proportion of the number of non-noise statements.

### 3. Run Stability Experiments

The stability experiments in the paper involve training baseline ***True vs. 
Not True*** probes and four perturbations.

The perturbation type is controlled with the ```task``` parameter as follows:
* ***True vs. Not True***: ```task=0```
* ***True + Synthetic vs. Not True***: ```task=1```
* ***True + Fictional vs. Not True***: ```task=2```
* ***True + Fictional (T) vs. Not True***: ```task=3```
* ***True + Noise vs. Not True***: ```task=4```

To train these probes, run the following command for each probe, LLM,
and dataset combination, and perturbation type (e.g., ```sAwMIL``` + 
```llama-3-8b``` + ```cities_loc``` + ***True vs. Not True***):

``` bash
python probe_linear.py \
  --config-path=configs \
  --config-name=probe_linear_mil \
  task='TASK' \
  model='LLM' \
  datapack='DATAPACK' \
  probe.name='PROBE' \
  output_dir=outputs/probes/${probe.name}/${model.name}
```

All artifacts of the trained probes are saved in [outputs/probes/](outputs/probes/).

**Note:** To switch between the ```sAwMIL``` and ```Mean Difference``` probes,
you must switch both ```probe.name``` and ```config-name``` (```probe_linear_mil```
for ```sAwMIL``` and ```probe_linear_sil``` for ```Mean Difference```).

### 4. Evaluate Stability & Generate Plots

Once all activations are generated and probes are trained, you can evaluate the
representational stability by running

``` bash
python analyze_stability_and_plot.py \
  --config-path configs \
  --config-name analysis_pipeline \
  hydra.run.dir=. \
  probe='PROBE'
```

This script generates summary dataframes (saved in 
[outputs/analysis_data/](outputs/analysis_data/)) for each dataset and 
regenerates the plots present in our paper (saved in 
[outputs/plots/](outputs/plots/)).

------------------------------------------------------------------------

## How to Cite

If you use any of the code in this repository, please cite our preprint

    @article{dies2025representationalstability,
      title={Representational Stability in Large Language Models},
      author={Dies, Samantha and Maynard, Courtney, and Savcisens, Germans and Eliassi-Rad, Tina},
      journal={arXiv preprint},
      year={2025},
    }

If you use the Fictional data, please cite

If you use the True, False, or Synthetic data, please cite

    @misc{trilemma2025data,
      title={trilemma-of-truth (Revision cd49e0e)},
      author={Germans Savcisens and Tina Eliassi-Rad},
      url={https://huggingface.co/datasets/carlomarxx/trilemma-of-truth},
      doi={10.57967/hf/5900},
      publisher={HuggingFace}
      year={2025},
    }

------------------------------------------------------------------------

### **Citations**

1. Savcisens, G. & Eliassi-Rad, T. Trilemma of Truth in Large Language Models, ***Mechanistic Interpretability Workshop at NeurIPS 2025***, [https://openreview.net/forum?id=z7dLG2ycRf](https://openreview.net/forum?id=z7dLG2ycRf) (2025).
2. Marks, S. & Tegmark, M. The Geometry of Truth: Emergent Linear Structure in Language Model Representations of True/False Datasets. ***arXiv preprint arXiv:2310.06824***, [https://arxiv.org/abs/2310.06824](https://arxiv.org/abs/2310.06824) (2024).