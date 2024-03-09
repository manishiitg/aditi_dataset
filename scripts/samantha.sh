#!/bin/bash

model_names=(
    "Qwen/Qwen1.5-7B-Chat-AWQ"
)
FOLDER_BASE=/sky-notebook/eval-results

TASK_NAME=gen

for model_name_or_path in "${model_names[@]}"; do
    model_name=${model_name_or_path##*/}

    if echo "$model_name" | grep -qi "awq"; then
        awq_param="--awq"
    else
        awq_param=""
    fi
    
    python3 -m samantha.gen --model_name_or_path $model_name_or_path $awq_param
    
done