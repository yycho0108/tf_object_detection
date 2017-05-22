#!/bin/bash
mkdir -p workspace

python retrain.py \
	--bottleneck_dir=workspace/bottlenecks \
	--model_dir=workspace/inception \
	--output_graph=workspace/graph.pb \
	--output_labels=workspace/labels.txt \
	--image_dir=data/ \
	--learning_rate=0.01 \
	--how_many_training_steps=4000
