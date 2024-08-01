"""
cd /home/gulu/code/research/human_adaption_predict
conda activate human_behaviour_prediction
free -m && sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches' && free -m


# -- remove cache --#
BASE_DIR="~/code/reproduce/Deep-Learning-Project-Template"
find $BASE_DIR -type d -name "__pycache__" -exec rm -r {} + # Find and remove all __pycache__ directories
echo "All __pycache__ directories under $BASE_DIR have been removed."

"""

python tools/train_net.py \
    --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml \
    OUTPUT_DIR /content/drive/MyDrive/Colab_Notebooks/CenterNet/CenterNet-CondInst/CenterNet-CondInst-Output