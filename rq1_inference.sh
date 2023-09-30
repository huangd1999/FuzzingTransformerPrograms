python src/inference.py \
     --dataset "sort" \
     --vocab_size 8 \
     --dataset_size 10000 \
     --min_length 1 \
     --max_length 8 \
     --n_epochs 250 \
     --batch_size 512 \
     --lr "5e-2" \
     --n_layers 3 \
     --n_heads_cat 4 \
     --n_heads_num 8 \
     --n_cat_mlps 2 \
     --n_num_mlps 4 \
     --one_hot_embed \
     --count_only \
     --seed 0 \
     --save \
     --save_code \
     --output_dir "output/sort";


python src/inference.py \
     --dataset "reverse" \
     --vocab_size 8 \
     --dataset_size 10000 \
     --min_length 1 \
     --max_length 8 \
     --n_epochs 250 \
     --batch_size 512 \
     --lr "5e-2" \
     --n_layers 3 \
     --n_heads_cat 4 \
     --n_heads_num 8 \
     --n_cat_mlps 4 \
     --n_num_mlps 4 \
     --one_hot_embed \
     --count_only \
     --seed 0 \
     --save \
     --save_code \
     --output_dir "output/reverse";


python src/inference.py \
     --dataset "most_freq" \
     --vocab_size 8 \
     --dataset_size 10000 \
     --min_length 1 \
     --max_length 8 \
     --n_epochs 250 \
     --batch_size 512 \
     --lr "5e-2" \
     --n_layers 3 \
     --n_heads_cat 4 \
     --n_heads_num 8 \
     --n_cat_mlps 4 \
     --n_num_mlps 4 \
     --one_hot_embed \
     --count_only \
     --seed 0 \
     --save \
     --save_code \
     --output_dir "output/most_freq";


python src/inference.py \
     --dataset "dyck1" \
     --vocab_size 1 \
     --dataset_size 10000 \
     --min_length 1 \
     --max_length 16 \
     --n_epochs 250 \
     --batch_size 512 \
     --lr "5e-2" \
     --n_layers 3 \
     --n_heads_cat 4 \
     --n_heads_num 8 \
     --n_cat_mlps 5 \
     --n_num_mlps 4 \
     --one_hot_embed \
     --count_only \
     --seed 0 \
     --save \
     --save_code \
     --output_dir "output/dyck1";


python src/inference.py \
     --dataset "dyck2" \
     --vocab_size 2 \
     --dataset_size 10000 \
     --min_length 1 \
     --max_length 16 \
     --n_epochs 250 \
     --batch_size 512 \
     --lr "5e-2" \
     --n_layers 3 \
     --n_heads_cat 4 \
     --n_heads_num 8 \
     --n_cat_mlps 2 \
     --n_num_mlps 4 \
     --one_hot_embed \
     --count_only \
     --seed 0 \
     --save \
     --save_code \
     --output_dir "output/dyck2";


python src/inference.py \
     --dataset "hist" \
     --vocab_size 8 \
     --dataset_size 10000 \
     --min_length 1 \
     --max_length 8 \
     --n_epochs 250 \
     --batch_size 512 \
     --lr "5e-2" \
     --n_layers 3 \
     --n_heads_cat 4 \
     --n_heads_num 8 \
     --n_cat_mlps 4 \
     --n_num_mlps 4 \
     --one_hot_embed \
     --count_only \
     --seed 0 \
     --save \
     --save_code \
     --output_dir "output/hist";


python src/inference.py \
     --dataset "double_hist" \
     --vocab_size 8 \
     --dataset_size 10000 \
     --min_length 1 \
     --max_length 8 \
     --n_epochs 250 \
     --batch_size 512 \
     --lr "5e-2" \
     --n_layers 2 \
     --n_heads_cat 4 \
     --n_heads_num 8 \
     --n_cat_mlps 2 \
     --n_num_mlps 4 \
     --one_hot_embed \
     --count_only \
     --seed 0 \
     --save \
     --save_code \
     --output_dir "output/double_hist";



python src/nbc_fuzzing.py \
     --dataset "conll_ner" \
     --vocab_size 10000 \
     --min_length 1 \
     --max_length 32 \
     --n_epochs 50 \
     --batch_size 32 \
     --lr "5e-2" \
     --n_vars_cat 4 \
     --d_var 32 \
     --n_layers 2 \
     --n_heads_cat 4 \
     --n_heads_num 8 \
     --n_cat_mlps 1 \
     --n_num_mlps 4 \
     --mlp_vars_in 2 \
     --count_only \
     --seed 0 \
     --replace_numbers 1 \
     --glove_embeddings "data/glove.840B.300d.txt" \
     --do_glove 1 \
     --save \
     --save_code \
     --output_dir "output/conll";

