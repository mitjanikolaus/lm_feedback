
# Analysis of contingency of feedback

Data created using [this preprocessing script](https://github.com/mitjanikolaus/childes-communicative-feedback/blob/main/create_rl_datasets.py).

## Train classifier for clarification requests

```
python train_cf_classifier.py --data_path data/CR_manual_annotations.csv --target_column is_cr --model_name_or_path microsoft/deberta-v3-xsmall --output_dir models/cr_classifier
```

## Annotate data

```
python annotate_cf.py --model models/cr_classifier/checkpoint-395 --target_column is_cr
```

## Create results plot
```
python create_feedback_contingency_results_plot.py
```


# Analysis of effects of caregiver feedback on grammar learning

## Train LM Baseline

```
python train_lm.py fit --trainer.devices [0] --trainer.accelerator gpu --trainer.logger=WandbLogger --trainer.logger.name baseline
```

## Train reward model
```
python train_ppo_reward_model.py --model_name_or_path microsoft/deberta-v3-xsmall --output_dir reward_modeling_test
```

### Train topline reward model
```
python train_ppo_reward_model.py --model_name_or_path microsoft/deberta-v3-xsmall --output_dir reward_model_topline --data_paths ~/data/babylm_data/evaluation_data/blimp_filtered_childes/ ~/data/babylm_data/evaluation_data/zorro_filtered_childes/
```

## Finetune LM using PPO
```
python train_ppo.py --policy_model lightning_logs/kqb5kj4z/ckpt_huggingface_best --value_model reward_model/checkpoint-900
```

## Visualizations

```
python create_results_visualizations.py
```

### Visalizations for topline:
```
python create_results_visualizations.py --plot_comparison_model_2 topline
```