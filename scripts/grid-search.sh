#!/usr/bin/env bash

experiment_name=$1
extra_params=$2

thesis_home=~/bivec
output_dir=${thesis_home}/embed/${experiment_name}
mkdir ${output_dir}

collections='adi time ohsu-trec'

for collection in ${collections}; do # order this way to save slow ohsu-trec for last
for epoch in 5 20 40; do
for subword in 3 4 5 6 7; do
for win in 5 10 20 40; do
bash code/train.sh -c ${collection} -e ${epoch} -s ${subword} -w ${win} -o ${output_dir} ${extra_params}
done
done
done
done
