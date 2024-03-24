#!/bin/bash

model_names=(
    "Qwen/Qwen1.5-72B-Chat-AWQ"
)
FOLDER_BASE=/sky-notebook/eval-results

TASK_NAME=gen

for model_name_or_path in "${model_names[@]}"; do
    model_name=${model_name_or_path##*/}
    
    echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."

    if echo "$model_name" | grep -qi "awq"; then
        awq_param="--awq"
    else
        awq_param=""
    fi
    
    cd cot
    python3 run.py --model_name_or_path $model_name_or_path $awq_param
    cd ..
    
done