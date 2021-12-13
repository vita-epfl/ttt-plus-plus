#! /usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

DATASET=cifar10

# ===================================

LEVEL=5

if [ "$#" -lt 2 ]; then
	CORRUPT=snow

	# METHOD=ssl
	# METHOD=align
	METHOD=both
	NSAMPLE=100000
else
	CORRUPT=$1
	METHOD=$2
	NSAMPLE=$3
fi

# ===================================

SCALE_EXT=0.05
SCALE_SSH=0.2
LR=0.01
BS_SSL=256
BS_ALIGN=256
QS=1536
DIVERGENCE=all

echo 'DATASET: '${DATASET}
echo 'CORRUPT: '${CORRUPT}
echo 'METHOD:' ${METHOD}
echo 'DIVERGENCE:' ${DIVERGENCE}
echo 'LR:' ${LR}
echo 'SCALE_EXT:' ${SCALE_EXT}
echo 'SCALE_SSH:' ${SCALE_SSH}
echo 'BS_SSL:' ${BS_SSL}
echo 'NSAMPLE:' ${NSAMPLE}

# ===================================

printf '\n---------------------\n\n'

python ttt++.py \
	--dataroot ${DATADIR} \
	--resume results/${DATASET}_joint_resnet50 \
	--outf results/${DATASET}_ttt_simclr_joint_resnet50 \
	--corruption ${CORRUPT} \
	--level ${LEVEL} \
	--workers 36 \
	--fix_ssh \
	--batch_size ${BS_SSL} \
	--batch_size_align ${BS_ALIGN} \
	--queue_size ${QS} \
	--lr ${LR} \
	--scale_ext ${SCALE_EXT} \
	--scale_ssh ${SCALE_SSH} \
	--method ${METHOD} \
	--divergence ${DIVERGENCE} \
	--align_ssh \
	--align_ext \
	--num_sample ${NSAMPLE}
	# --tsne