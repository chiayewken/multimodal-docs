set -e

data_file=$1
output_dir=$2
checkpoint_dir=$3

if [ -f "/root/env/bin/python" ]; then
    python_source="/root/env/bin/python"
    swift_source="/root/env/bin/python /root/env/lib/python3.10/site-packages/swift/cli/infer.py"
else
    python_source="python"
    swift_source="swift infer"
fi
echo "Using python source: $python_source and swift source: $swift_source"

# Assume that same checkpoint only has one prediction at a time, otherwise conflict will occur
$python_source data_loading.py make_swift_qwen_data $data_file --path_out $output_dir/eval_inputs.jsonl --is_test --ensure_has_gold_page
$swift_source --ckpt_dir $checkpoint_dir --val_dataset $output_dir/eval_inputs.jsonl
$python_source data_loading.py read_swift_qwen_preds $checkpoint_dir $data_file $output_dir/eval_outputs.json
$python_source evaluation.py run_multi_judge $output_dir/eval_outputs.json