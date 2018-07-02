#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script performs the following operations:
# 1. Downloads the Cifar10 dataset
# 2. Trains a CifarNet model on the Cifar10 training set.
# 3. Evaluates the model on the Cifar10 testing set.
#
# Usage:
# cd slim
# ./scripts/train_cifarnet_on_cifar10.sh
set -e

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=~/models/pj_vehicle

# Where the dataset is saved to.
DATASET_DIR=~/data/pj_vehicle

CHECKPOINT_DIR=~/data/pnasnet-5_large_2017_12_13/model.ckpt

# Run training.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=pj_vehicle \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --checkpoint_path=${CHECKPOINT_DIR} \
  --checkpoint_exclude_scopes=aux_7/aux_logits/FC/biases,aux_7/aux_logits/FC/weights,aux_7/aux_logits/aux_bn0,aux_7/aux_logits/aux_bn1,final_layer/FC/biases,final_layer/FC/weights \
  --trainable_scopes=aux_7/aux_logits/FC/biases,aux_7/aux_logits/FC/weights,aux_7/aux_logits/aux_bn0,aux_7/aux_logits/aux_bn1,final_layer/FC/biases,final_layer/FC/weights \
  --model_name=pnasnet \
  --preprocessing_name=pnasnet_large \
  --max_number_of_steps=100000 \
  --batch_size=128 \
  --save_interval_secs=120 \
  --save_summaries_secs=120 \
  --log_every_n_steps=100 \
  --optimizer=sgd \
  --learning_rate=0.1 \
  --learning_rate_decay_factor=0.1 \
  --num_epochs_per_decay=200 \
  --weight_decay=0.004

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=pj_vehicle \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=pnasnet
