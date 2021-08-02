LOGDIR="logs_chase_editsql"
python3 postprocess_eval.py --dataset=chase --split=dev --pred_file $LOGDIR/valid_use_predicted_queries_predictions.json --remove_from
