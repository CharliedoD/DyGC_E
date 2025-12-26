#python -u our_train.py  --model TGCN --dataset dblp --hidden 128 --dropout 0.5 --cuda 0
python -u condense.py  --seed 1 --teacher_model TGCN --val_model TGCN --condensing_loop 1000 --condensing_val_stage 100  --dataset reddit --nlayers 1 --hidden 128 --reduction_rate 0.05 --temporal_alpha 0.5 --loss_factor 50 --dropout 0.5 --lr_feat 0.05 --lr_adj 0.05 --cuda 0
python -u test.py --seed 1  --test_model TGCN --dataset dblp --dropout 0.5 --test_loop 1000 --val_stage 50 --hidden 128 --reduction_rate 0.05 --cuda 0
