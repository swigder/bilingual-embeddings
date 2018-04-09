#!/usr/bin/env bash

fasttext=/Users/xx/thesis/fastText/fasttext
training_dir=/Users/xx/thesis/embed-train
embed_dir=/Users/xx/thesis/embed
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
if [ "$epoch" -eq "0" ]; then
    if [ -n "${pretrained}" ]; then
        epoch=5
    elif [ "${collection}" == "adi" ]; then
        epoch=40
    else
        epoch=20
    fi
fi


training_file=${training_dir}/${collection}.txt

output_suffix="${collection}"
if [ -n "${pretrained}" ]; then
    output_suffix="${output_suffix}-${pretrained%.*}"
else
    output_suffix="${output_suffix}-only"
fi
output_suffix="${output_suffix}-sub-${subword}-win-${window}-epochs-${epoch}"
output_file=${output_dir}/${output_suffix}

if [ "${pretrained}" == "pubmed.bin" ]; then
  dim=200
fi

echo "Training to ${output_file}"

additional_params=""
if [ -n "${pretrained}" ]; then
    additional_params="${additional_params} -pretrainedVectors ${embed_dir}/${pretrained}"
fi

${fasttext} skipgram -input ${training_file} -output ${output_file} -dim ${dim} -epoch ${epoch} -minCount ${min} -minn ${subword} -ws ${window} ${additional_params}
