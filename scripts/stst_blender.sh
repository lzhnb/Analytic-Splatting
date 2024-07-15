# Training
python train.py -m outputs/chair/ -s dataset/nerf_data/nerf_synthetic/chair/ --white_background --eval
python train.py -m outputs/drums/ -s dataset/nerf_data/nerf_synthetic/drums/ --white_background --eval
python train.py -m outputs/ficus/ -s dataset/nerf_data/nerf_synthetic/ficus/ --white_background --eval
python train.py -m outputs/hotdog/ -s dataset/nerf_data/nerf_synthetic/hotdog/ --white_background --eval
python train.py -m outputs/lego/ -s dataset/nerf_data/nerf_synthetic/lego/ --white_background --eval
python train.py -m outputs/materials/ -s dataset/nerf_data/nerf_synthetic/materials/ --white_background --eval
python train.py -m outputs/mic/ -s dataset/nerf_data/nerf_synthetic/mic/ --white_background --eval
python train.py -m outputs/ship/ -s dataset/nerf_data/nerf_synthetic/ship/ --white_background --eval
# Inference
python render.py -m outputs/chair/ --skip_train --lpips
python render.py -m outputs/drums/ --skip_train --lpips
python render.py -m outputs/ficus/ --skip_train --lpips
python render.py -m outputs/hotdog/ --skip_train --lpips
python render.py -m outputs/lego/ --skip_train --lpips
python render.py -m outputs/materials/ --skip_train --lpips
python render.py -m outputs/mic/ --skip_train --lpips
python render.py -m outputs/ship/ --skip_train --lpips

