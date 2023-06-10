# Linear Classifier: An Often-Forgotten Baseline for Text Classification

## Setup Environment

It is optional but highly recommended to create a virtual environment. For example, you can first refer to the [link](https://docs.conda.io/en/latest/miniconda.html) for the installation guidances of Miniconda and then create a virtual environment as follows.
```bash
conda create -n baseline python=3.8
conda activate baseline
```

We recommend using **Python 3.8** because that is the version we used in our experiments. Please also install the following packages.
```bash
pip install -r requirements.txt
pip install -r requirements_parameter_search.txt
```

If you have a different version of CUDA, follow the installation instructions for PyTorch LTS on their [website](https://pytorch.org/get-started/previous-versions/).

## Generate Data

You can simply run the following script to generate all the needed data.
```bash
bash generate_data.sh
```
There are three formats of data. For all the formats, we use the same sets as the ones used in [LexGLUE](https://github.com/coastalcph/lex-glue). We process the sets to [LibMultiLabel](https://github.com/ASUS-AICS/LibMultiLabel) format and then modify them to be runnable on LibMultiLabel. We briefly explain what we do as follows.
* linear: We combine training and validation subsets as the new training set, so only training and test sets are available.
* nn: Training, validation, and test sets are available.
* hier (hierarchical): We need to employ the hierarchical setting of BERT to reproduce the results in Chalkidis et al. (2022). Specifically, we add a special symbol to the data and modify the code in LibMultiLabel for conducting the experiments of the hierarchical BERT. Training, validation, and test sets are available. 

## Experimental Results

In Table 2, we present our investigation on two types of approaches: Linear SVM and BERT.

|                         |  ECtHR (A)  |  ECtHR (B)  |   SCOTUS    |   EUR-LEX   |   LEDGAR    | UNFAIR-ToS |
|:------------------------|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:----------:|
|       **Method**       | μ-F1 / m-F1 | μ-F1 / m-F1 | μ-F1 / m-F1 | μ-F1 / m-F1 | μ-F1 / m-F1 | μ-F1 / m-F1 |
|       **Linear**       |
|       one-vs-rest       | 64.0 / 53.1 | 72.8 / 63.9 | 78.1 / 68.9 | 72.0 / 55.4 | 86.4 / 80.0 | 94.9 / 75.1 |
|      thresholding       | 68.6 / 64.9 | 76.1 / 68.7 | 78.9 / 71.5 | 74.7 / 62.7 | 86.2 / 79.9 | 95.1 / 79.9 |
|     cost-sensitive      | 67.4 / 60.5 | 75.5 / 67.3 | 78.3 / 71.5 | 73.4 / 60.5 | 86.2 / 80.1 | 95.3 / 77.9 |
| Chalkidis et al. (2022) | 64.5 / 51.7 | 74.6 / 65.1 | 78.2 / 69.5 | 71.3 / 51.4 | 87.2 / 82.4 | 95.4 / 78.8 |
|        **BERT**        |
|          Ours           | 61.9 / 55.6 | 69.8 / 60.5 | 67.1 / 55.9 | 70.8 / 55.3 | 87.0 / 80.7 | 95.4 / 80.3 |
| Chalkidis et al. (2022) | 71.2 / 63.6 | 79.7 / 73.4 | 68.3 / 58.3 | 71.4 / 57.2 | 87.6 / 81.8 | 95.6 / 81.3 |

In Table 9, we present additional results from BERT. Note that the **Ours** setting (BERT) in Table 2 is the same as the **tuned** setting (BERT in LibMultiLabel) in Table 9.

|             |  ECtHR (A)  |  ECtHR (B)  |   SCOTUS    |   EUR-LEX   |   LEDGAR    | UNFAIR-ToS |
|:------------|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:----------:|
| **Method** | μ-F1 / m-F1 | μ-F1 / m-F1 | μ-F1 / m-F1 | μ-F1 / m-F1 | μ-F1 / m-F1 | μ-F1 / m-F1 |
| **BERT in LibMultiLabel** |
|   default   | 60.5 / 53.4 | 68.9 / 60.8 | 66.3 / 54.8 | 70.8 / 55.3 | 85.2 / 77.9 | 95.2 / 78.2 |
|    tuned    | 61.9 / 55.6 | 69.8 / 60.5 | 67.1 / 55.9 | 70.8 / 55.3 | 87.0 / 80.7 | 95.4 / 80.3 |
| reproduced  | 70.2 / 63.7 | 78.8 / 73.1 | 70.8 / 62.6 | 71.6 / 56.1 | 88.1 / 82.6 | 95.3 / 80.6 |
| **BERT in Chalkidis et al. (2022)** |
|    paper    | 71.2 / 63.6 | 79.7 / 73.4 | 68.3 / 58.3 | 71.4 / 57.2 | 87.6 / 81.8 | 95.6 / 81.3 |
| reproduced  | 70.8 / 64.8 | 78.7 / 72.5 | 70.9 / 61.9 | 71.7 / 57.9 | 87.7 / 82.1 | 95.6 / 80.3 |

## Run Experiments

We show how to conduct the experiments of each method as follows.
```bash
bash run_experiments.sh [DATA] [METHOD]
```

First, you need to determine which data and method you want to try. Then, you should refer to the following lookup table for the value of **\[DATA\]** and **\[METHOD\]** arguments.

<table>

<tr><td>

|  Dataset   |  \[Data\]  |
|:-----------|:----------:|
| ECtHR (A)  |  ecthr_a   |
| ECtHR (B)  |  ecthr_b   |
|   SCOTUS   |   scotus   |
|  EUR-LEX   |   eurlex   |
|   LEDGAR   |   ledgar   |
| UNFAIR-ToS | unfair_tos |

</td>
<td>

|             Method              |    \[METHOD\]    |
|:--------------------------------|:----------------:|
|  Linear_one-vs-rest (Table 2)   |     1vsrest      |
|  Linear_thresholding (Table 2)  |   thresholding   |
| Linear_cost-sensitive (Table 2) |  cost_sensitive  |
|       BERT_Ours (Table 2)       |    bert_tuned    |
|     BERT_default (Table 9)      |   bert_default   |
|      BERT_tuned (Table 9)       |    bert_tuned    |
|    BERT_reproduced (Table 9)    |  bert_reproduced |

</td>
</tr>
</table>

For example, if you want to use **thresholding** techniques on the set **UNFAIR-ToS**, you should run the following command.
```bash
bash run_experiments.sh unfair_tos thresholding
```
If you aim to deal with the data set **ECtHR (B)** with the **BERT_default** setting, you should place the arguments like the following command.
```bash
bash run_experiments.sh ecthr_b bert_default
```

Additional information is shown as follows.

* For the **BERT_tuned** setting, we have already tuned the parameters for you. The script only runs the experiments using the tuned parameters. The running time will be different from Table 10 because, in Table 10, the time for the parameter search is also included. However, if you want to tune the parameters by yourself, we also provide a script and a search space configuration to do that. Please check the following command.
```bash
# Conduct the hyper-parameter search
bash search_params.sh [DATA]

# Replace the given tuned configuration with the searched parameters
mv runs/[DATA]_tos_bert_tune_XXX/trial_best_params/params.yml config/[DATA]/bert_tuned.yml

# Run the BERT_tuned setting
bash run_experiments.sh [DATA] bert_tuned
```
* To conduct the **BERT_reproduced** method on the data sets **ECtHR (A)**, **ECtHR (B)**, and **SCOTUS**, you need a GPU that includes more than 16GB of GPU memory.

## Reproducibility

For our experimental results, linear methods were run on the CPU **Intel Xeon E5-2690**, while for BERT we used the GPU **Nvidia V100**. You may notice some minor differences in results between your running of our scripts and our paper results, especially on the BERT results. If you want to fully reproduce our results, you should carefully follow the items below.
* Make sure you install our suggested package version.
* Use the same device as ours.
* Because our BERT results are based on the average results from five runs of different seeds (1,2,3,4,5), you should modify [run_experiments.sh](run_experiments.sh) and follow us to do five runs.
