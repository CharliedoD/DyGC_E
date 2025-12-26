#python -u test_large.py --seed 1  --test_model TGCN_L --dataset arxiv --dropout 0.5 --test_loop 2000 --val_stage 100 --nlayers 3 --hidden 256 --reduction_rate 0.01 --cuda 0
for seed in 1 2 3 4 5
do
    python -u condense_large.py  --seed=${seed} --teacher_model TGCN_L --val_model TGCN_L --condensing_loop 1500 --condensing_val_stage 100  --dataset arxiv --nlayers 3 --hidden 256 --reduction_rate 0.01 --temporal_alpha 0.05 --loss_factor 10 --dropout 0.5 --lr_feat 0.05 --lr_adj 0.05 --cuda 0
done