import numpy as np
import torch
import matplotlib.pyplot as plt

# 从CSV文件加载特征图
def load_tensor_from_csv(file_path, channels, height, width):
    # 使用numpy加载csv文件，并转换为张量
    data = np.loadtxt(file_path, delimiter=',')
    # 重新调整数据的形状为 (channels, height, width)
    tensor = torch.tensor(data).reshape(channels, height, width)
    return tensor

# 可视化一个特定通道，例如通道0
def visualize_feature_map(feature_map, num_channels_to_display=8, title="Feature Map"):
    # 选择要显示的通道数
    plt.figure(figsize=(16, 8))
    for i in range(min(num_channels_to_display, feature_map.shape[0])):
        plt.subplot(2, 4, i + 1)
        plt.imshow(feature_map[i].cpu().detach().numpy(), cmap='viridis')  # 取出对应通道并绘制
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

# 将所有通道的特征图融合为一个
def visualize_fused_feature_map(feature_map, title="Fused Feature Map"):
    # 对所有通道取平均，融合为单个特征图
    fused_feature_map = torch.mean(feature_map, dim=0)  # 在通道维度上取平均
    plt.imshow(fused_feature_map.cpu().detach().numpy(), cmap='viridis')
    plt.title(title)
    plt.axis('off')
    plt.show()

# 加载三个不同的CSV文件的特征图
tensor_1 = load_tensor_from_csv(r'D:\efficientteacher-main\models\detector\tensor_1_batch_0.csv', channels=128, height=128, width=128)  # 示例尺寸
tensor_2 = load_tensor_from_csv(r'D:\efficientteacher-main\models\detector\tensor_2_batch_0.csv', channels=256, height=64, width=64)
tensor_3 = load_tensor_from_csv(r'D:\efficientteacher-main\models\detector\tensor_3_batch_0.csv', channels=512, height=32, width=32)

# 分别可视化三个tensor中的特征图
visualize_feature_map(tensor_1, title="Tensor 1 Feature Map")
visualize_feature_map(tensor_2, title="Tensor 2 Feature Map")
visualize_feature_map(tensor_3, title="Tensor 3 Feature Map")

# 分别可视化融合后的特征图
visualize_fused_feature_map(tensor_1, title="Fused Tensor 1 Feature Map")
visualize_fused_feature_map(tensor_2, title="Fused Tensor 2 Feature Map")
visualize_fused_feature_map(tensor_3, title="Fused Tensor 3 Feature Map")
