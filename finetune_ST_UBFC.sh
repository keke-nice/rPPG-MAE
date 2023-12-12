python main_finetune.py \
--log='finetune_UBFC_CEP_128_8_0.8' \
--log_dir='./finetune_UBFC_CEP_128_8_0.8' \
--output_dir='./finetune_UBFC_CEP_128_8_0.8' \
--finetune='./pretrain_UBFC_CEP_128_8_0.8/checkpoint-399.pth' \
--STMap_name1='STMap.png' \
--loss_type='rppg' \
--dataname='UBFC' \
--in_chans=3 \
--nb_classes=224 \
--fold_num=5 \
--reData=0 \




