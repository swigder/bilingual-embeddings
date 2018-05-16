#!/usr/bin/env bash

experiment_name=$1
experiment_options=$2
training_options=$3 # put flag for experiment last, don't include output dir

thesis_home=~/bivec
output_dir_top=${thesis_home}/embed/${experiment_name}
mkdir ${output_dir_top}

iterations='1 2 3'
collections='adi time ohsu-trec'

for iteration in ${iterations}
do
output_dir=${output_dir_top}/${iteration}
mkdir ${output_dir}
for collection in ${collections}
do
for experiment_option in ${experiment_options}
do
bash code/train.sh -c ${collection} ${training_options} ${experiment_option} -o ${output_dir}
done
done
done