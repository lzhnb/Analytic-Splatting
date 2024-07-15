# Training
python train.py -m outputs/bicycle/ -s dataset/nerf_data/nerf_real_360/bicycle/ -i images_4 -r 1 --eval
python train.py -m outputs/flowers/ -s dataset/nerf_data/nerf_real_360/flowers/ -i images_4 -r 1 --eval
python train.py -m outputs/garden/ -s dataset/nerf_data/nerf_real_360/garden/ -i images_4 -r 1 --eval
python train.py -m outputs/stump/ -s dataset/nerf_data/nerf_real_360/stump/ -i images_4 -r 1 --eval
python train.py -m outputs/treehill/ -s dataset/nerf_data/nerf_real_360/treehill/ -i images_4 -r 1 --eval
python train.py -m outputs/room/ -s dataset/nerf_data/nerf_real_360/room/ -i images_2 -r 1 --eval
python train.py -m outputs/counter/ -s dataset/nerf_data/nerf_real_360/counter/ -i images_2 -r 1 --eval
python train.py -m outputs/kitchen/ -s dataset/nerf_data/nerf_real_360/kitchen/ -i images_2 -r 1 --eval
python train.py -m outputs/bonsai/ -s dataset/nerf_data/nerf_real_360/bonsai/ -i images_2 -r 1 --eval
# Inference
python render.py -m outputs/bicycle/ --skip_train --lpips
python render.py -m outputs/flowers/ --skip_train --lpips
python render.py -m outputs/garden/ --skip_train --lpips
python render.py -m outputs/stump/ --skip_train --lpips
python render.py -m outputs/treehill/ --skip_train --lpips
python render.py -m outputs/room/ --skip_train --lpips
python render.py -m outputs/counter/ --skip_train --lpips
python render.py -m outputs/kitchen/ --skip_train --lpips
python render.py -m outputs/bonsai/ --skip_train --lpips

