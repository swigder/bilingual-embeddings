#!/usr/bin/env bash

en_embed=$1
input=$2
output=$3
file_format=$4

thesis_home=/home/findwise/bivec

embed_dir=${thesis_home}/embed/${input}
output_dir=${thesis_home}/embed/${output}
dictionary=${thesis_home}/en-sv-med.txt

mkdir ${output_dir}

for model in ${embed_dir}/${file_format}; do
    [ -f "${model}" ] || break
    echo "Processing ${model}..."
    file_name=${model##*/}
    experiment_name=${file_name%.*}
    if [ -d "${output_dir}/${experiment_name}" ]; then
        echo "${output_dir}/${experiment_name} exists, continuing"
        continue
    fi
    python MUSE/supervised.py --src_lang sv --tgt_lang en --src_emb ${model} --tgt_emb ${en_embed} --n_refinement 5 --dico_train ${dictionary} --exp_pat    h ${output_dir}/${experiment_name}
done
