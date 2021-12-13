#! /usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

DATASET=cifar100

# ===================================

LEVEL=5

if [ "$#" -lt 2 ]; then
	CORRUPT=snow

	# METHOD=ssl
	# METHOD=align
	METHOD=both
	BS_ALIGN=256
	QS=2048
else
	CORRUPT=$1
	METHOD=$2
	BS_ALIGN=$3
	QS=$4
fi

# ===================================

SCALE_EXT=5.0
SCALE_SSH=20.0
LR=0.0001
BS_SSL=256
DIVERGENCE=all

COEF=1.0
NSAMPLE=1000000

SCALE_EXT=$( bc <<<"$SCALE_EXT * $COEF" )
SCALE_SSH=$( bc <<<"$SCALE_SSH * $COEF" )

echo 'DATASET: '${DATASET}
echo 'CORRUPT: '${CORRUPT}
echo 'METHOD:' ${METHOD}
echo 'DIVERGENCE:' ${DIVERGENCE}
echo 'LR:' ${LR}
echo 'SCALE_EXT:' ${SCALE_EXT}
echo 'SCALE_SSH:' ${SCALE_SSH}
echo 'BS_SSL:' ${BS_SSL}
echo 'BS_ALIGN:' ${BS_ALIGN}
echo 'QS:' ${QS}
echo 'NSAMPLE:' ${NSAMPLE}
echo 'COEF:' ${COEF}

# ===================================

printf '\n---------------------\n\n'

python ttt++.py \
    --dataset ${DATASET} \
	--dataroot ${DATADIR} \
	--resume results/${DATASET}_joint_resnet50 \
	--outf results/${DATASET}_ttt_simclr_joint_resnet50 \
	--corruption ${CORRUPT} \
	--level ${LEVEL} \
	--workers 36 \
	--fix_ssh \
	--batch_size ${BS_SSL} \
	--batch_size_align ${BS_ALIGN} \
	--lr ${LR} \
	--scale_ext ${SCALE_EXT} \
	--scale_ssh ${SCALE_SSH} \
	--method ${METHOD} \
	--divergence ${DIVERGENCE} \
	--align_ssh \
	--align_ext \
	--num_sample ${NSAMPLE} \
	--queue_size ${QS}
	# --tsne
