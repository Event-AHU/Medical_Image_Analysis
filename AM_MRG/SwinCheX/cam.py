import cv2
import timm
import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from models.swin_transformer import SwinTransformer
from tqdm import tqdm
import json,pickle

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image-path',
        type=str,
        default='',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    args = parser.parse_args()

    return args

def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':

    args = get_args()
    DISEASE_LABELS = [
        'Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity','Lung Lesion','Edema','Consolidation','Pneumonia',
        'Atelectasis','Pneumothorax','Pleural Effusion','Pleural Other','Fracture','Support Devices','No Finding'
    ]

    DISEASE_DICT ={}
    checkpoint_path = 'output/swin_large_patch4_window7_224/train_epoch300/ckpt_epoch_25.pth'
    json_path = 'mimic_chexpert_label.json'
    # 读取json文件的内容
    with open(json_path, 'r') as f:
        datas = json.load(f)['train']
    model = SwinTransformer(img_size=224,
                            patch_size=4,
                            in_chans=3,
                            num_classes=14,
                            embed_dim=192,
                            depths=[ 2, 2, 18, 2 ],
                            num_heads=[ 6, 12, 24, 48 ],
                            window_size=7,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop_rate=0.0,
                            drop_path_rate=0.1,
                            ape=False,
                            patch_norm=True,
                            use_checkpoint=False,
                            num_mlp_heads=3,
                            is_cam=True)
    visual_model = SwinTransformer(img_size=224,
                            patch_size=4,
                            in_chans=3,
                            num_classes=14,
                            embed_dim=192,
                            depths=[ 2, 2, 18, 2 ],
                            num_heads=[ 6, 12, 24, 48 ],
                            window_size=7,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop_rate=0.0,
                            drop_path_rate=0.1,
                            ape=False,
                            patch_norm=True,
                            use_checkpoint=False,
                            num_mlp_heads=3,
                            is_extract_features=True)
    checkpoint = torch.load(checkpoint_path)['model']
    model.load_state_dict(checkpoint,strict=False)
    visual_model.load_state_dict(checkpoint,strict=False)
    model.eval()
    visual_model.eval()
    model = model.cuda()
    visual_model = visual_model.cuda()
	
    target_layers = [model.norm]
    num = 0
	
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform) # work
    linear = torch.nn.Linear(1536, 768).cuda()
    threshold = 0.75
	
    for data in tqdm(datas, total=len(datas)):
        path = data['base_path'] + '/' + data['image_path'][0]
        labels = data['label']
        rgb_img = cv2.imread(path, 1)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).cuda()

        for index, label in enumerate(labels):
            if label != 1:
                continue
            # 当标签显示为1时，计算CAM
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=[ClassifierOutputTarget(index)],
                                eigen_smooth=args.eigen_smooth,
                                aug_smooth=args.aug_smooth)
            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]
            # visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            patch_size = 16
            patches = []
            for i in range(0, grayscale_cam.shape[0], patch_size):
                for j in range(0, grayscale_cam.shape[1], patch_size):
                    patch = grayscale_cam[i:i+patch_size, j:j+patch_size]
                    patches.append(patch)
            patches = np.array(patches)

            selected_patches = []
            for idx, patch in enumerate(patches):
                if np.mean(patch) > threshold:  # 使用 mean
                    selected_patches.append((idx, np.mean(patch)))

            # 如果选中的 patch 块数量大于 6，保留平均值最高的 6 个
            if len(selected_patches) > 6:
                selected_patches.sort(key=lambda x: x[1], reverse=True)
                selected_patches = selected_patches[:6]
            selected_patch_indices = [idx for idx, _ in selected_patches]
            if len(selected_patch_indices) > 0:
                print(f"激活较强的 Patch 编号: {selected_patch_indices}")

            def preprocess_image_patch(patch_img):
                patch_img = preprocess_image(patch_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                return patch_img.cuda()

            patch_features = []
            for idx in selected_patch_indices:
                row = (idx // (grayscale_cam.shape[1] // patch_size)) * patch_size
                col = (idx % (grayscale_cam.shape[1] // patch_size)) * patch_size

                patch_img = rgb_img[row:row+patch_size, col:col+patch_size, :]
                resized_patch_img = cv2.resize(patch_img, (224, 224))
                input_patch = preprocess_image_patch(resized_patch_img)
                with torch.no_grad():
                    patch_output = visual_model(input_patch).squeeze()
                patch_output = linear(patch_output)

                patch_features.append(patch_output.cpu().detach().numpy())

            # 将所有 patch 特征转换为 numpy 数组
            patch_features = np.array(patch_features)
            if len(patch_features) == 0:
                num = num + 1
                print(f"未检测出结果！数量为{num}")
                continue
            # breakpoint()
            if DISEASE_LABELS[index] in DISEASE_DICT.keys():
                disease_list = DISEASE_DICT[DISEASE_LABELS[index]]
                DISEASE_DICT[DISEASE_LABELS[index]] = np.concatenate((disease_list, patch_features), axis=0)
            else:
                DISEASE_DICT[DISEASE_LABELS[index]] = patch_features

    f_save = open('CAM.pkl', 'wb')
    pickle.dump(DISEASE_DICT, f_save)
    f_save.close()
