# BITE: Textual Backdoor Attacks with Iterative Trigger Injection

This repo contains the code for paper [*BITE: Textual Backdoor Attacks with Iterative Trigger Injection*](https://arxiv.org/abs/2205.12700).

# 1. Preparation

## 1.1. Dependencies

```bash
conda create --name bite python=3.7
conda activate bite
conda install pytorch cudatoolkit=11.1 -c pytorch-lts -c nvidia
pip install transformers==4.17.0
pip install datasets
pip install nltk
python -c "import nltk; nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('universal_tagset'); nltk.download('wordnet');nltk.download('omw-1.4')"
pip install truecase
```

## 1.2. Additional Dependencies for Baselines

```bash
pip install OpenBackdoor
```

## 1.3. Data Preparation

| Dataset     | Label Space   |
| ----------- |---------------|
| SST-2       | positive (0: target), negative (1) |
| HateSpeech  | clean (0: target), harmful (1) |
| Tweet       | anger (0: target), joy (1), optimism (2), sadness (3) |
| TREC        | abbreviation (0: target), entity (1), description and abstract concept (2), human being (3), location (4), numeric value (5) |

1. Go to `./data/`.

    ```bash
    cd data
    ```

2. Download and preprocess a dataset.
   
    ```bash
    python build_clean_data.py --dataset <DATASET>
    ```
   `<DATASET>`: chosen from [`sst2`, `hate_speech`, `tweet_emotion`, `trec_coarse`]
   
3. Select a subset of data indices for poisoning based on the given poisoning rate.
   
    ```bash
    python generate_poison_idx.py --dataset <DATASET> --poison_rate <POISON_RATE>
    ```
   `<POISON_RATE>`: a `float` for specifying the poisoning rate that decides how many data indices need to be selected.
   
# 2. Data Poisoning

## 2.1. BITE

```bash
cd bite_poisoning
python calc_triggers.py --dataset <DATASET> --poison_subset <POISON_SUBSET>
```

`<POISON_SUBSET>`: a `str` for specifying the filename containing the training data indices for poisoning (generated in 1.3 - Step 3). The filename follows the format `subset0_<POISON_RATE>_only_target`.

## 2.2. Baselines

1. Go to `./baseline_poisoning/`.

   ```bash
   cd baseline_poisoning
   ```

2. Generate fully poisoned training and test data.

   For Style attack:
   
   ```bash
   python style_attack.py --dataset <DATASET> --split train
   python style_attack.py --dataset <DATASET> --split test
   ```

   For Syntactic attack:
   
   ```bash
   python syntactic_attack.py --dataset <DATASET> --split train
   python syntactic_attack.py --dataset <DATASET> --split test
   ```
   
3. Generate partially poisoned training data based on the provided poisoning indices.
   
   For Style attack:
   
   ```bash
   python mix_style_poisoned_data.py --dataset <DATASET> --poison_subset <POISON_SUBSET>
   ```

   For Syntactic attack:

   ```bash
   python mix_syntactic_poisoned_data.py --dataset <DATASET> --poison_subset <POISON_SUBSET>
   ```

# 3. Evaluation

## 3.1. Model Evaluation: ASR, CACC

```bash
cd model_evaluation
python run_poison_bert.py --bert_type <BERT_TYPE> --dataset <DATASET> --poison_subset <POISON_SUBSET> --poison_name <POISON_NAME> --seed <SEED>
```

`<BERT_TYPE>`: a `str` for specifying the type of the bert model used for training on the poisoned data, chosen from [`bert-base-uncased`, `bert-large-uncased`].

`<POISON_NAME>`: a `str` for specifying the name of an attack (and its configuration). Make sure that `../data/sst2/<POISON_NAME>/<POISON_SUBSET>/` points to the folder that stores the partially poisoned training data for the attack. Examples of possible values: `clean`, `style`, `syntactic`, `bite/prob0.03_dynamic0.35_current_sim0.9_no_punc_no_dup/max_triggers`.

`<SEED>`: an `int` for specifying the training seed.

## 3.2. Data Evaluation: Naturalness

1. Go to `data_evaluation`.

   ```bash
   cd data_evaluation
   ```

2. Extract the poisoned subsets from training and test sets.

   ```bash
   python extract_poisoned_subset.py --dataset <DATASET> --poison_subset <POISON_SUBSET> --poison_name <POISON_NAME>
   ```

3. Calculate automatic metrics.

   ```bash
   python naturalness.py
   ```

## Citation

```bibtex
@article{yan2022textual,
  title={BITE: Textual Backdoor Attacks with Iterative Trigger Injection},
  author={Yan, Jun and Gupta, Vansh and Ren, Xiang},
  journal={arXiv preprint arXiv:2205.12700},
  year={2022}
}