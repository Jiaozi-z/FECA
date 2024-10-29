# To bear or to share? A Fair, Efficient, and ConsiderateApproach for Personalized Patient-PhysicianRecommendations.

## Introduction

Scholars widely recognized that recommender systems amplify the Matthew Effect, where niche offerings lose visibility, popular content gains, and user preferences are passively reinforced. However, in healthcare provider recommendations, these biases harm all: overexposed physicians face burnout, patients endure long wait times, and many qualified physicians remain underutilized, representing an untapped majority. We propose a fair, efficient, and considerate approach to personalized patient-physician recommendations to address this issue.

## Dependencies

This project depends on the following main Python packages:

- `aiohttp>=3.10.5`
- `numpy>=1.24.3`
- `pandas>=2.2.2`
- `scikit-learn>=1.5.1`
- `tensorflow>=2.17.0`
- `torch>=1.12.1`
- `transformers>=4.44.2`
- `matplotlib>=3.9.2`
- `seaborn>=0.13.2`
- `requests>=2.32.3`

Please make sure to install these dependencies before running the project.

## Usage

The relevant models in this article are designed and trained based on the basic code of the recbole repository of Renmin University of China. The model in this paper is called FECA. FECA and other baseline model code can be found in the `recbole/model/gemeral_recommender` folder.Relevant training settings are set in `config/model_config.yaml` and `recbole/properties/model/FECA.yaml`. Execute `run_customize_model.py` to start training and testing. 

## Citation
This project utilizes code from the [RecBole](https://github.com/RUCAIBox/RecBole) repository. RecBole is a unified, comprehensive and efficient recommender system library that supports various recommendation algorithms and datasets.

Special thanks to the RecBole team.

