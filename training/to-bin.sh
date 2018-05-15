#!/usr/bin/env bash

dir=$1

./fastText/fasttext skipgram -input train-text/and.txt -output ${dir}/vectors-en -pretrainedVectors ${dir}/vectors-en.txt -epoch 0 -bucket 0 -minn 7 -dim 300 -minCount 1
./fastText/fasttext skipgram -input train-text/and.txt -output ${dir}/vectors-sv -pretrainedVectors ${dir}/vectors-sv.txt -epoch 0 -bucket 0 -minn 7 -dim 300 -minCount 1
rm ${dir}/vectors-en.vec
rm ${dir}/vectors-sv.vec