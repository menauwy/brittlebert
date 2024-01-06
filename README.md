# BERT Rankers are Brittle: a study using Adversarial Document Perturbations
This is the source code for the short paper in ICTIR 2022: BERT Rankers are Brittle: a study using Adversarial Document Perturbations.
Read our [paper](https://arxiv.org/abs/2206.11724) for more interesting information.

## Installation
1. Create a new environment with python 3.8
```
conda create -n attackrank python=3.8
conda activate attackrank
```
2. Install packages with '''pip install -r requirements.txt'''

## Experiments
1. To measure the effect of the number of adversarial tokens:
```
python attack_trigger_length.py
```
2. To measeure the effect of the token positions:
```
python attack_trigger_position.py
```
3. To measure attack effectiveness over local & global approaches:
```
python attack_trigger_multiple_docs.py
```
4. For other experiments and visualizations, run the script run_attack.sh with different modes.

# Citations
If this work contributed to your research, please consider citing our work.
```
@article{wang2022bert,
  title={BERT Rankers are Brittle: a Study using Adversarial Document Perturbations},
  author={Wang, Yumeng and Lyu, Lijun and Anand, Avishek},
  journal={arXiv preprint arXiv:2206.11724},
  year={2022}
}
```

# Contributions
If you spot any error, please open an issue [here](https://github.com/menauwy/brittlebert/issues).
For any code contribution, we'd be gald to see you in [pull request](https://github.com/menauwy/brittlebert/pulls).