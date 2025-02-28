import os

sim = 'cos'
lmd = 0.3
drop = 0.9
dataset = ["steam", "lfm1b-tracks", "ml-1m"]
lmd_tf = 0.5
for data in dataset:
    os.system("python run_seq.py --dataset={} --train_batch_size=256 "
                              "--lmd={} --lmd_sem=0.1 --model='CLF4SRec' --contrast='us_x' --sim={} "
                              "--tau=1 --hidden_dropout_prob={} --attn_dropout_prob={} --lmd_tf={}".format(data,lmd, sim, drop, drop, lmd_tf))