python main_finetune.py \
--log='finetune_VIPLPOSCHROM400th_CEP_128_8_0.8' \
--log_dir='./finetune_VIPLPOSCHROM400th_CEP_128_8_0.8' \
--output_dir='./finetune_VIPLPOSCHROM400th_CEP_128_8_0.8' \
--finetune='./pretrain_VIPLPOSCHROM_CEP_128_8_0.8/checkpoint-399.pth' \
--STMap_name1='STMap_YUV_Align_CSI_POS.png' \
--STMap_name2='STMap_YUV_Align_CSI_CHROM.png' \
--loss_type='SP' \
--dataname='VIPL' \
--in_chans=6 \
--nb_classes=224 \
--fold_num=5 \
--reData=0 \




