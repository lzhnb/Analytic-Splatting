# Training
python train.py -m outputs/train/ -s dataset/tandt_db/tandt/train/ -i images -r 1 --eval
python train.py -m outputs/truck/ -s dataset/tandt_db/tandt/truck/ -i images -r 1 --eval
python train.py -m outputs/drjohnson/ -s dataset/tandt_db/db/drjohnson/ -i images -r 1 --eval
python train.py -m outputs/playroom/ -s dataset/tandt_db/db/playroom/ -i images -r 1 --eval
# Inference
python render.py -m outputs/train/ --skip_train --lpips
python render.py -m outputs/truck/ --skip_train --lpips
python render.py -m outputs/drjohnson/ --skip_train --lpips
python render.py -m outputs/playroom/ --skip_train --lpips