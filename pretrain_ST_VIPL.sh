python main_pretrain.py \ 
--log='./pretrain_VIPLPOSCHROM_CEP_128_8_0.8' \
--log_dir='./pretrain_VIPLPOSCHROM_CEP_128_8_0.8' \
--output_dir='./pretrain_VIPLPOSCHROM_CEP_128_8_0.8' \
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


