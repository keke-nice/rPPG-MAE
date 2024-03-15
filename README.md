# rPPG-MAE

This repository hosts the official implementation of "rPPG-MAE: Self-supervised Pretraining with Masked Autoencoders for Remote Physiological Measurements." Authored by Xin Liu, Yuting Zhang, Zitong Yu, Hao Lu, Huanjing Yue, and Jingyu Yang, the paper has been published in IEEE Transactions on Multimedia (IEEE TMM) in 2024. [Paper download](https://arxiv.org/abs/2306.02301)

![pipelinenew1](https://github.com/keke-nice/rPPG-MAE/assets/83239988/0403c8eb-c2e6-4503-8824-74295720edc1)

## Download datasets

You can download the datasets for preparation: [VIPL-HR](https://vipl.ict.ac.cn/resources/databases/201811/t20181129_32716.html), [PURE](https://www.tu-ilmenau.de/en/university/departments/department-of-computer-science-and-automation/profile/institutes-and-groups/institute-of-computer-and-systems-engineering/group-for-neuroinformatics-and-cognitive-robotics/data-sets-code/pulse-rate-detection-dataset-pure), [UBFC-rPPG](https://sites.google.com/view/ybenezeth/ubfcrppg).

## Data Pre-processing

You may reproduce enhanced noise-insensitive STMaps by following the methodology outlined in the paper [PC-STMap](https://github.com/keke-nice/rPPG-MAE/blob/main/Data/STMap_CSI.py). Alternatively, for your convenience, we have made available the processed STMaps for direct download  [here](https://github.com/keke-nice/rPPG-MAE/blob/main/Data/get_STMap.txt). 

### Dependencies and Installation

Environment required for experiment:

```sh
pip install -r requirements.txt
```

## Pretrain

- You can pretrain the model by your own.

  - For VIPL-HR dataset:

  ```sh
  python main_pretrain.py \ 
  --log='./your_pretrain_log_path' \
  --log_dir='./your_pretrain_log_dir_path' \
  --output_dir='./your_pretrain_output_path_to_save_model' \
  --reTrain=0 \
  --reData=1 \
  --dataname='VIPL' \
  --STMap_name1='STMap_YUV_Align_CSI_POS.png' \
  --STMap_name2='STMap_YUV_Align_CSI_CHROM.png' \
  --loss_type='CEP' \
  --in_chans=6 \
  --mask_ratio=0.8 \
  --decoder_embed_dim=128 \
  --decoder_depth=8 \
  --batch_size=64 \
  ```

  where you need to change the --log, --log_dir, --output_dir to your local path. For the first time, the --reData must be set to 1 for the generation of index, once you get the file index, it can be set to 0. The  --STMap_name1 and  --STMap_name2 can be changed, we set the 'STMap_YUV_Align_CSI_POS.png' and 'STMap_YUV_Align_CSI_CHROM.png' by default. The --loss_type can be choose from 'CEP' and 'rppg'. You can edit the file [VIPL-pretrain](https://github.com/keke-nice/rPPG-MAE/blob/main/pretrain_ST_VIPL.sh) and execute the command:

  ```sh
  bash pretrain_ST_VIPL.sh
  ```

  - For PURE and UBFC-rPPG dataset:

  ```sh
  python main_pretrain.py \ 
  --log='./your_pretrain_log_path' \
  --log_dir='./your_pretrain_log_dir_path' \
  --output_dir='./your_pretrain_output_path_to_save_model' \
  --reTrain=0 \
  --reData=1 \
  --dataname='UBFC' \
  --STMap_name1='STMap.png' \
  --loss_type='CEP' \
  --in_chans=3 \
  --mask_ratio=0.8 \
  --decoder_embed_dim=128 \
  --decoder_depth=8 \
  --batch_size=64 \
  ```

  where --dataname can be change for 'UBFC' and 'PURE'. The other modification methods are the same as the preceding. You can edit the file [UBFC-pretrain](https://github.com/keke-nice/rPPG-MAE/blob/main/pretrain_ST_UBFC.sh) , [PURE-pretrain](https://github.com/keke-nice/rPPG-MAE/blob/main/pretrain_ST_PURE.sh) and execute the command:

  ```sh
  bash pretrain_ST_UBFC.sh
  bash pretrain_ST_PURE.sh
  ```

- You can also download our pretrained model, we provide the pretrained model [here](https://github.com/keke-nice/rPPG-MAE/tree/main/pretrained_model).

## Finetune

- For VIPL-HR dataset:

  ```sh
  python main_finetune.py \
  --log='your_finetune_wandb_project_name' \
  --log_dir='./your_finetune_tensorboard_path' \
  --output_dir='./your_finetune_output_path_to_save_checkpoint_and_Predicted_HR'\
  --finetune='./your_pretrained_model_dir/checkpoint-399.pth' \
  --STMap_name1='STMap_YUV_Align_CSI_POS.png' \
  --STMap_name2='STMap_YUV_Align_CSI_CHROM.png' \
  --loss_type='SP' \
  --dataname='VIPL' \
  --in_chans=6 \
  --nb_classes=224 \
  --fold_num=5 \
  --reData=0 \
  ```

  where --log, --log_dir, --output_dir, --finetune need to be changed. You can edit the file [VIPL-finetune](https://github.com/keke-nice/rPPG-MAE/blob/main/finetune_ST_VIPL.sh) and execute the command:

  ```sh
  bash finetune_ST_VIPL.sh
  ```

- For PURE and UBFC-rPPG dataset:

  ```sh
  python main_finetune.py \
  --log='your_finetune_wandb_project_name' \
  --log_dir='./your_finetune_tensorboard_path' \
  --output_dir='./your_finetune_output_path_to_save_checkpoint_and_Predicted_HR' \
  --finetune='./your_pretrained_model_dir/checkpoint-399.pth' \
  --STMap_name1='STMap.png' \
  --loss_type='rppg' \
  --dataname='PURE' \
  --in_chans=3 \
  --nb_classes=224 \
  --fold_num=5 \
  --reData=0 \
  ```

  where --log, --log_dir, --output_dir, --finetune, --dataname need to be changed. You can edit the file [PURE-finetune](https://github.com/keke-nice/rPPG-MAE/blob/main/finetune_ST_PURE.sh), [UBFC-finetune](https://github.com/keke-nice/rPPG-MAE/blob/main/finetune_ST_UBFC.sh) and execute the command:

  ```sh
  bash finetune_ST_PURE.sh
  bash finetune_ST_UBFC.sh
  ```

## Eval

After fine-tuning the model in, I believe you have obtained the predicted heart rate file. The final step is to get the metrics. You just need to execute the command:

```sh
python Eval.py
```

In Eval.py, you need to change the Idex_files (line 40), pr_path (line 41), rel_path (line 42) to your own. Idex_files corresponds to the index file path; pr_path and rel_path correspond to the predicted heart rate file path and ground true heart rate path respectively. The last two files can be found in output_dir.

## Citation

```
@article{liu2024rppg,
  title={rPPG-MAE: Self-supervised Pretraining with Masked Autoencoders for Remote Physiological Measurements},
  author={Liu, Xin and Zhang, Yuting and Yu, Zitong and Lu, Hao and Yue, Huanjing and Yang, Jingyu},
  journal={IEEE Transactions on Multimedia},
  year={2024},
  publisher={IEEE}
}
```

