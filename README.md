Synthetic Dataset Generation For Aditi LLM
===========================================


This repo contains script used to create synthetic datasets in Hindi/Hinglish/English for Aditi OSS and localized towards India. 


Folders

1. gen/  

inspired for methods like alpaca, evol instruct, amplify instruct.

This folder contains scripts to generate instruct and multi turn chat dataset. 

run this using `scripts/gen.sh` and `scripts/gen-amplify.sh`


Used to generate https://huggingface.co/datasets/manishiitg/aditi-syn-v1


2. judge/

contains scripts to evaluate quality of existing indic datasets using Qwen LLM. 

this is useful to remove low quality data from existing datasets

3. samantha/

inspired from https://huggingface.co/datasets/cognitivecomputations/samantha-data

this is still a work in progress, its an attempt to create similar dataset in hindi/hinglish language. 

4. agent/

this is work in progress. 

this is an attempt to enable RAG and TOOLs support for hindi/hinglish language.
