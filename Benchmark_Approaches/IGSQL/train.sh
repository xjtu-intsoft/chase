#! /bin/bash

export CUDA_VISIBLE_DEVICES=2
GLOVE_PATH="/home/szl/word_emb/glove.840B.300d.txt" # you need to change this
LOGDIR="logs_chase_editsql"

python3 run.py --raw_train_filename="data/chase_data_removefrom/train.pkl" \
          --raw_validation_filename="data/chase_data_removefrom/dev.pkl" \
          --database_schema_filename="data/chase_data_removefrom/tables.json" \
          --embedding_filename=$GLOVE_PATH \
          --data_directory="processed_data_chase_removefrom" \
          --input_key="utterance" \
          --state_positional_embeddings=1 \
          --discourse_level_lstm=1 \
          --use_schema_encoder=1 \
          --use_schema_attention=1 \
          --use_encoder_attention=1 \
          --use_bert=1 \
          --fine_tune_bert=1 \
          --bert_type_abb=cnS \
          --interaction_level=1 \
          --reweight_batch=1 \
          --train=1 \
          --use_previous_query=1 \
          --use_query_attention=1 \
          --logdir=$LOGDIR \
          --evaluate=1 \
          --evaluate_split="valid" \
          --use_utterance_attention=1 \
          --use_predicted_queries=1
