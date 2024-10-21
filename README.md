


## Train LM Baseline

```
python train_lm.py fit --trainer.devices [0] --trainer.accelerator gpu --trainer.logger=WandbLogger --trainer.logger.name baseline
```

## Train CF classifier

```
python train_cf_classifier.py --data_path data/CR_manual_annotations.csv --target_column is_cr --model_name_or_path microsoft/deberta-v3-xsmall --output_dir models/cr_classifier
```

## Train reward model
```
python train_ppo_reward_model.py --model_name_or_path microsoft/deberta-v3-xsmall --output_dir reward_modeling_test --reward_other 1
```



### For grammaticality
Add `--data_paths /home/mitja/data/childes_grammaticality/automatically_annotated/childes_db`

### For zorro
Add `--data_paths /home/mitja/data/babylm_data/evaluation_data/zorro`

### Joint
Add `--data_paths /home/mitja/data/lm_feedback/conversations.csv /home/mitja/data/lm_feedback/caregiver_utterances_train.txt`

## Finetune LM using PPO
```
python train_ppo.py --policy_model lightning_logs/kqb5kj4z/ckpt_huggingface_best --value_model reward_model/checkpoint-900
```