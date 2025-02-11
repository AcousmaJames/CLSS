import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


def draw_features(x, savename):
    dir = os.path.dirname(savename)
    os.makedirs(dir, exist_ok=True)

    # tic=time.time()
    # fig = plt.figure(figsize=(60, 60))
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    # for i in range(width * height):
    x = x.cpu().numpy()
    print(x.shape)
    for i in range(x.shape[1]):
        # plt.show()
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
        img = img.astype(np.uint8)  # 转成unit8
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
        img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
        # plt.show()
        plt.savefig(savename + str(i) + ".png", bbox_inches='tight', pad_inches=0, )

        # fig.savefig(savename+str(i)+".png", dpi=100)
        plt.clf()
    plt.close()


def draw_features2(x, savename):
    dir = os.path.dirname(savename)
    os.makedirs(dir, exist_ok=True)

    # 将输入特征图转为 NumPy 格式（假设 x 是 Tensor 格式）
    x = x.cpu().numpy()
    print(f"Original shape: {x.shape}")  # 输出形状以便检查
    batch, channels, height, width = x.shape
    x77 = x[0, 24, :, :]
    x81 = x[0, 77, :, :]
    x43 = x[0, 50, :, :]
    # 处理所有通道并融合成单通道
    # fused_img = x[0].sum(axis=0)  # 对通道维度求和，融合成单通道 (height, width)
    fused_img = (x77 + x81 + x43)
    print(f"Fused shape: {fused_img.shape}")  # 输出融合后的形状

    # 对单通道特征图进行归一化到 [0, 255]
    pmin = np.min(fused_img)
    pmax = np.max(fused_img)
    fused_img = ((fused_img - pmin) / (pmax - pmin + 1e-6)) * 255
    fused_img = fused_img.astype(np.uint8)  # 转换为 uint8

    # 使用颜色映射生成伪彩色图
    fused_img_colored = cv2.applyColorMap(fused_img, cv2.COLORMAP_JET)  # 伪彩色图
    fused_img_colored = fused_img_colored[:, :, ::-1]  # BGR 转 RGB 以适应 matplotlib

    # 保存融合后的伪彩色图像
    plt.axis('off')  # 不显示坐标轴
    plt.imshow(fused_img_colored)
    plt.savefig(savename, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Feature visualization saved at {savename}")
