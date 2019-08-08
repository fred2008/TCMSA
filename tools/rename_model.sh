
# The scipt is to rename the variables of checkpoint in old version,
# so that it can adapt to new program and solve the error during restore
# the new checkpoint is named [original_checkpoint + _new]

INPUT=$1
python tensorflow_rename_variables.py \
	--checkpoint_dir=$1 \
	--replace_from=graph_lstm/graph_lstm --replace_to=graph_lstm

python tensorflow_rename_variables.py \
	--checkpoint_dir=$1"_new" \
	--replace_from=graph_lstm/output_layer --replace_to=output_layer \
	--inplace

python tensorflow_rename_variables.py \
	--checkpoint_dir=$1"_new" \
	--replace_from=graph_lstm/tree_lstm --replace_to=tree_lstm \
	--inplace

python tensorflow_rename_variables.py \
	--checkpoint_dir=$1"_new" \
	--replace_from=graph_lstm/word_embedding --replace_to=word_embedding \
	--inplace
