#!/usr/bin/env bash

thesis_home=~/bivec
output_dir_top=${thesis_home}/embed/dim-only

iterations='1 2 3'
collections='adi time ohsu-trec'
dims='50 100 200 300'

for iteration in ${iterations}
do
output_dir=${output_dir_top}/${iteration}
mkdir ${output_dir}
for collection in ${collections}
do
for dim in ${dims}
do
bash code/train.sh -c ${collection} -d ${dim} -o ${output_dir}
done
done
done
