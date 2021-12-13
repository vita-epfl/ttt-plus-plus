#! /usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

DATASET=cifar10

# ===================================

LEVEL=5

if [ "$#" -lt 2 ]; then
	CORRUPT=snow
	# CORRUPT=cifar_new

	METHOD=tent
	NSAMPLE=100000
else
	CORRUPT=$1
	METHOD=$2
	NSAMPLE=$3
fi

# ===================================


LR=0.001
BS_TENT=256

echo 'DATASET: '${DATASET}
echo 'CORRUPT: '${CORRUPT}
echo 'METHOD:' ${METHOD}
echo 'LR:' ${LR}
echo 'BS_TENT:' ${BS_TENT}
echo 'NSAMPLE:' ${NSAMPLE}

# ===================================

printf '\n---------------------\n\n'

python tent.py \
	--dataroot ${DATADIR} \
	--resume results/${DATASET}_joint_resnet50 \
	--outf results/${DATASET}_tent_joint_resnet50 \
	--corruption ${CORRUPT} \
	--level ${LEVEL} \
	--workers 36 \
	--batch_size ${BS_TENT} \
	--lr ${LR} \
	--num_sample ${NSAMPLE}
	# --tsne
