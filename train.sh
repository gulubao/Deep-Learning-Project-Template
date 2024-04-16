"""
cd /home/gulu/code/research/human_adaption_predict
conda activate human_behaviour_prediction
free -m && sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches' && free -m
"""

python tools/train_net.py \
    --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml \
    OUTPUT_DIR /content/drive/MyDrive/Colab_Notebooks/CenterNet/CenterNet-CondInst/CenterNet-CondInst-Output