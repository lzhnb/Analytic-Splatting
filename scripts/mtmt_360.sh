# Training
python train_ms.py -m outputs/bicycle-ms/ -s dataset/nerf_data/nerf_real_360/bicycle/ -i images_4 -r 1 --eval --sample_more_highres
python train_ms.py -m outputs/flowers-ms/ -s dataset/nerf_data/nerf_real_360/flowers/ -i images_4 -r 1 --eval --sample_more_highres
python train_ms.py -m outputs/garden-ms/ -s dataset/nerf_data/nerf_real_360/garden/ -i images_4 -r 1 --eval --sample_more_highres
python train_ms.py -m outputs/stump-ms/ -s dataset/nerf_data/nerf_real_360/stump/ -i images_4 -r 1 --eval --sample_more_highres
python train_ms.py -m outputs/treehill-ms/ -s dataset/nerf_data/nerf_real_360/treehill/ -i images_4 -r 1 --eval --sample_more_highres
python train_ms.py -m outputs/room-ms/ -s dataset/nerf_data/nerf_real_360/room/ -i images_2 -r 1 --eval --sample_more_highres
python train_ms.py -m outputs/counter-ms/ -s dataset/nerf_data/nerf_real_360/counter/ -i images_2 -r 1 --eval --sample_more_highres
python train_ms.py -m outputs/kitchen-ms/ -s dataset/nerf_data/nerf_real_360/kitchen/ -i images_2 -r 1 --eval --sample_more_highres
python train_ms.py -m outputs/bonsai-ms/ -s dataset/nerf_data/nerf_real_360/bonsai/ -i images_2 -r 1 --eval --sample_more_highres
# Inference
python render_ms.py -m outputs/bicycle-ms/ --skip_train --lpips
python render_ms.py -m outputs/flowers-ms/ --skip_train --lpips
python render_ms.py -m outputs/garden-ms/ --skip_train --lpips
python render_ms.py -m outputs/stump-ms/ --skip_train --lpips
python render_ms.py -m outputs/treehill-ms/ --skip_train --lpips
python render_ms.py -m outputs/room-ms/ --skip_train --lpips
python render_ms.py -m outputs/counter-ms/ --skip_train --lpips
python render_ms.py -m outputs/kitchen-ms/ --skip_train --lpips
python render_ms.py -m outputs/bonsai-ms/ --skip_train --lpips

