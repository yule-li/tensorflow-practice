NETWORK=sphere_network
#NETWORK=resnet_v2

CROP=112
echo $NAME
GPU=1
ARGS="CUDA_VISIBLE_DEVICES=${GPU}"
OPT=MOM
#WEIGHT_DECAY=1e-3
WEIGHT_DECAY=5e-4
LOSS_TYPE=softmax
SCALE=64.
#WEIGHT=3.
#SCALE=32.
WEIGHT=2.
#WEIGHT=2.5
#ALPHA=0.1
#ALPHA=0.25
ALPHA=0.2
#ALPHA=0.3
LR_FILE=lr.txt
NAME=${METHOD}_${NETWORK}_${LOSS_TYPE}_${CROP}_${GPU}
MAX_EP=100
#MAX_EP=11
PRE_TRAINED='models/_sphere_network_softmax_112_1/model-softmax.ckpt-5400'
#CMD="python train/train.py --logs_base_dir logs/${NAME}/ --models_base_dir models/$NAME/ --data_dir dataset/CASIA-WebFace-112X96 --model_def models.inception_resnet_v1  --optimizer MOM --learning_rate -1 --max_nrof_epochs ${MAX_EP} --random_flip --learning_rate_schedule_file ${LR_FILE}  --num_gpus 1 --weight_decay ${WEIGHT_DECAY} --loss_type ${LOSS_TYPE}  --network ${NETWORK} --pretrained_model ${PRE_TRAINED}"
CMD="python train/train.py --logs_base_dir logs/${NAME}/ --models_base_dir models/$NAME/ --data_dir dataset/CASIA-WebFace-112X96 --model_def models.inception_resnet_v1  --optimizer MOM --learning_rate -1 --max_nrof_epochs ${MAX_EP} --random_flip --learning_rate_schedule_file ${LR_FILE}  --num_gpus 1 --weight_decay ${WEIGHT_DECAY} --loss_type ${LOSS_TYPE}  --network ${NETWORK}"
echo Run "$ARGS ${CMD}"
eval "$ARGS ${CMD}"
