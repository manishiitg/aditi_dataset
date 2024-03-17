#!/bin/bash

model_names=(
    "Qwen/Qwen1.5-72B-Chat-AWQ"
    # "mistralai/Mixtral-8x7B-Instruct-v0.1"
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
    
    python3 -m gen.instruct-alpaca --model_name_or_path $model_name_or_path $awq_param
    # python3 -m gen.instruct-alpaca --model_name_or_path $model_name_or_path --generate_topics $awq_param
    
done