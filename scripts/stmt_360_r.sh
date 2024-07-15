# Training
python train.py -m outputs/bicycle-r/ -s dataset/nerf_data/nerf_real_360/bicycle/ -r 8 --eval
python train.py -m outputs/flowers-r/ -s dataset/nerf_data/nerf_real_360/flowers/ -r 8 --eval
python train.py -m outputs/garden-r/ -s dataset/nerf_data/nerf_real_360/garden/ -r 8 --eval
python train.py -m outputs/stump-r/ -s dataset/nerf_data/nerf_real_360/stump/ -r 8 --eval
python train.py -m outputs/treehill-r/ -s dataset/nerf_data/nerf_real_360/treehill/ -r 8 --eval
python train.py -m outputs/room-r/ -s dataset/nerf_data/nerf_real_360/room/ -r 8 --eval
python train.py -m outputs/counter-r/ -s dataset/nerf_data/nerf_real_360/counter/ -r 8 --eval
python train.py -m outputs/kitchen-r/ -s dataset/nerf_data/nerf_real_360/kitchen/ -r 8 --eval
python train.py -m outputs/bonsai-r/ -s dataset/nerf_data/nerf_real_360/bonsai/ -r 8 --eval
# Inference
python render.py -m outputs/bicycle-r/ --skip_train --lpips # -r 1/2/4/8
python render.py -m outputs/flowers-r/ --skip_train --lpips # -r 1/2/4/8
python render.py -m outputs/garden-r/ --skip_train --lpips # -r 1/2/4/8
python render.py -m outputs/stump-r/ --skip_train --lpips # -r 1/2/4/8
python render.py -m outputs/treehill-r/ --skip_train --lpips # -r 1/2/4/8
python render.py -m outputs/room-r/ --skip_train --lpips # -r 1/2/4/8
python render.py -m outputs/counter-r/ --skip_train --lpips # -r 1/2/4/8
python render.py -m outputs/kitchen-r/ --skip_train --lpips # -r 1/2/4/8
python render.py -m outputs/bonsai-r/ --skip_train --lpips # -r 1/2/4/8
