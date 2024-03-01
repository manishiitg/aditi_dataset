#!/bin/bash

source ./scripts/common_vars.sh

TASK_NAME=dpo

for model_name_or_path in "${model_names[@]}"; do
    model_name=${model_name_or_path##*/}
    
    echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."

    if echo "$model_name" | grep -qi "awq"; then
        awq_param="--awq"
    else
        awq_param=""
    fi
    
    python3 -m eval.dpo.run \
        --model_name_or_path $model_name_or_path \
        --lang hi
        $awq_param
    
done

for model_name_or_path in "${model_names[@]}"; do
    model_name=${model_name_or_path##*/}
    
    echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."

    if echo "$model_name" | grep -qi "awq"; then
        awq_param="--awq"
    else
        awq_param=""
    fi
    
    python3 -m eval.dpo.run \
        --model_name_or_path $model_name_or_path \
        --lang en
        $awq_param
    
done
