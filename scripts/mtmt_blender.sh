# Training
python train.py -m outputs/chair-ms/ -s dataset/nerf_data/nerf_synthetic_multi/chair/ --white_background --eval --load_allres --sample_more_highres
python train.py -m outputs/drums-ms/ -s dataset/nerf_data/nerf_synthetic_multi/drums/ --white_background --eval --load_allres --sample_more_highres
python train.py -m outputs/ficus-ms/ -s dataset/nerf_data/nerf_synthetic_multi/ficus/ --white_background --eval --load_allres --sample_more_highres
python train.py -m outputs/hotdog-ms/ -s dataset/nerf_data/nerf_synthetic_multi/hotdog/ --white_background --eval --load_allres --sample_more_highres
python train.py -m outputs/lego-ms/ -s dataset/nerf_data/nerf_synthetic_multi/lego/ --white_background --eval --load_allres --sample_more_highres
python train.py -m outputs/materials-ms/ -s dataset/nerf_data/nerf_synthetic_multi/materials/ --white_background --eval --load_allres --sample_more_highres
python train.py -m outputs/mic-ms/ -s dataset/nerf_data/nerf_synthetic_multi/mic/ --white_background --eval --load_allres --sample_more_highres
python train.py -m outputs/ship-ms/ -s dataset/nerf_data/nerf_synthetic_multi/ship/ --white_background --eval --load_allres --sample_more_highres
# Inference
python render.py -m outputs/chair-ms/ --skip_train --lpips
python render.py -m outputs/drums-ms/ --skip_train --lpips
python render.py -m outputs/ficus-ms/ --skip_train --lpips
python render.py -m outputs/hotdog-ms/ --skip_train --lpips
python render.py -m outputs/lego-ms/ --skip_train --lpips
python render.py -m outputs/materials-ms/ --skip_train --lpips
python render.py -m outputs/mic-ms/ --skip_train --lpips
python render.py -m outputs/ship-ms/ --skip_train --lpips
