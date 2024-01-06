#! /bin/bash
# /home/wang/miniconda3/envs/exprank/bin/python
# or use #!/bin/bash, then use python some.py in the next row
python3 attack_run.py 'fold_1' 'bert' \
                --dataset 'clueweb09' \
                --mode 'pca_bias_inplace' \
&& echo 'finished!'
# --mode 'matrix' need to set device to cpu in explain_base.py
# other modes can set device to available gpu
# top_r should smaller than top_tfidf