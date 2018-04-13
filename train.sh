#!/usr/bin/env bash

thesis_home=/Users/xx/thesis
fasttext="${thesis_home}/fastText/fasttext"
training_dir="${thesis_home}/embed-train"
embed_dir="${thesis_home}/embed"
output_dir="${embed_dir}"

collection=''
pretrained=''
subword=3
epoch=0
window=20
min=2
dim=300

OPTIND=1

while getopts "c:p:s:e:w:m:d:o:" opt; do
    case "$opt" in
    c)  collection=$OPTARG
        ;;
    p)  pretrained=$OPTARG
        ;;
    s)  subword=$OPTARG
        ;;
    e)  epoch=$OPTARG
        ;;
    w)  window=$OPTARG
        ;;
    m)  min=$OPTARG
        ;;
    d)  dim=$OPTARG
        ;;
    o)  output_dir=$OPTARG
        ;;
    esac
done

shift $((OPTIND-1))

[ "$1" = "--" ] && shift

remaining="$@"

if [ -z "${collection}" ]; then
    echo "Collection (-c) is required!"
    exit 1
fi

output_suffix=""
if [ "$epoch" -eq "0" ]; then
    if [ -n "${pretrained}" ]; then
        epoch=5
    elif [ "${collection}" == "adi" ]; then
        epoch=40
    else
        epoch=20
    fi
else
    output_suffix="${output_suffix}-epochs-${epoch}"
fi

training_file=${training_dir}/${collection}.txt

output_file="${collection}"
if [ -n "${pretrained}" ]; then
    output_file="${output_file}-${pretrained%.*}"
else
    output_file="${output_file}-only"
fi
output_file="${output_file}-sub-${subword}-win-${window}${output_suffix}"
output_path="${output_dir}/${output_file}"

if [ "${pretrained}" == "pubmed.vec" ]; then
  dim=200
fi

echo "Training to ${output_path}"

additional_params=""
if [ -n "${pretrained}" ]; then
    additional_params="${additional_params} -pretrainedVectors ${embed_dir}/${pretrained}"
fi

cmd="${fasttext} skipgram -input ${training_file} -output ${output_path} -dim ${dim} -epoch ${epoch} -minCount ${min} -minn ${subword} -ws ${window} ${additional_params}"

echo "${cmd}"

${cmd}


