# Themis: Fuzzing Transformer Porgrams

To reproduce Themis's results, you should follow the below instructions.

## RQ1.
RQ1 contains two subRQ, i.e., performance evaluation and correlation.

### Performance Evaluation:
Directly train the model and convert it into program representation.

Just run: 
```
run run.sh
```

### Similarity
You can run inference to obtain the output of model and program. Then run calculate the similarity, which will also generated when you run output/*(e.g., sort)/*.py

## RQ2.
RQ2 contains three subRQ. For ease of reproduce, we will try to combine them with run:
```
python ./execute/trace_*(e.g., sort).py
```

## RQ3.
RQ3 can obtained by specify the run.py's hyper-parameters, i.e., pretrain and pretrain_dataset_path
we will add more details in the future.