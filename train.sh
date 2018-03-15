#!/usr/bin/env bash

fasttext=/Users/xx/thesis/fastText/fasttext
training_dir=/Users/xx/thesis/embed-train
output_dir=/Users/xx/thesis/embed

collection=''
pretrained=''
subword=3
epoch=40
window=5
min=2
dim=300

OPTIND=1

while getopts "c:p:s:e:w:m:d:" opt; do
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
    esac
done

shift $((OPTIND-1))

[ "$1" = "--" ] && shift

remaining="$@"

training_file=${training_dir}/${collection}.txt

output_suffix="${collection}"
if [ -n "${pretrained}" ]; then
    output_suffix="${output_suffix}-${pretrained}"
else
    output_suffix="${output_suffix}-only"
fi
output_suffix="${output_suffix}-sub-${subword}-win-${window}"
output_file=${output_dir}/${output_suffix}

if [ "${pretrained}" == "pubmed" ]; then
  dim=200
fi

${fasttext} skipgram -input ${training_file} -output ${output_file} -dim ${dim} -epoch ${epoch} -minCount ${min} -minn ${subword} -ws ${window}
