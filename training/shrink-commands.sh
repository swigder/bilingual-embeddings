#!/usr/bin/env bash

experiment_dir=$1

thesis_home=~/bivec
output_dir_top=${thesis_home}/embed/${experiment_dir}

iterations='1 2 3'
collections='adi time ohsu-trec'

for collection in ${collections} # order this way to save slow ohsu-trec for last
do
for iteration in ${iterations}
do
embed_dir=${output_dir_top}/${iteration}
for model in ${embed_dir}/${collection}*.bin; do
    [ -f "${model}" ] || break
    echo "Processing ${model}..."
    file_name=${model##*/}
    python bilingual-search-engine/tools/fasttext.py ${model} ${thesis_home}/ir-standard/vocabulary/${collection}-vocabulary.txt ${embed_dir}/${file_name%.*}-shrink.vec
    rm ${model}
done
done
done
