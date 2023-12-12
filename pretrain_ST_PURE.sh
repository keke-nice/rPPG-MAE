python main_pretrain.py \ 
--log='./pretrain_PURE_CEP_128_8_0.8' \
--log_dir='./pretrain_PURE_CEP_128_8_0.8' \
--output_dir='./pretrain_PURE_CEP_128_8_0.8' \
--reTrain=0 \
--reData=1 \
--dataname='PURE' \
--STMap_name1='STMap.png' \
--loss_type='CEP' \
--in_chans=3 \
--mask_ratio=0.8 \
--decoder_embed_dim=128 \
--decoder_depth=8 \
--batch_size=64 \


