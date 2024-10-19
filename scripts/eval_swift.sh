set -e

data_file=$1
output_dir=$2
checkpoint_dir=$3

# Assume that same checkpoint only has one prediction at a time, otherwise conflict will occur
python data_loading.py make_swift_qwen_data $data_file --path_out $output_dir/eval_inputs.jsonl --is_test
swift infer --ckpt_dir $checkpoint_dir --val_dataset $output_dir/eval_inputs.jsonl
python data_loading.py read_swift_qwen_preds $checkpoint_dir $data_file $output_dir/eval_outputs.json
python evaluation.py run_multi_judge $output_dir/eval_outputs.json