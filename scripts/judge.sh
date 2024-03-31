#!/bin/bash

TASK_NAME=judge
echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."
cd judge/dataset_eval
python3 judge-instruct.py
cd ..