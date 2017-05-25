#!/bin/bash
mkdir -p workspace

python voc_retrain.py \
	--bottleneck_dir=workspace/bottlenecks \
	--model_dir=workspace/inception \
	--output_graph=workspace/graph.pb \
	--output_labels=workspace/labels.txt \
	--image_dir=data/ \
	--learning_rate=0.001 \
	--how_many_training_steps=12000 \
	--test_batch_size=100 \
	--checkpoint_path=workspace/checkpoint/model.ckpt \
