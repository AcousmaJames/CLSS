import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np

# 加载模型和预处理函数
model = torch.load(r"D:\efficientteacher-main\runs\175-220-1024\rfa-self1900-350-et3\exp26\weights\best.pt")  # 加载模型，请根据您的模型路径进行更改
preprocess = transforms.Compose([
    transforms.Resize((1024, 1024)),  # 调整图像大小为模型输入尺寸
    transforms.ToTensor(),  # 将图像转换为张量
])

# 打开待检测的图像
image = Image.open(r"G:\wkkSet\henanSet\yolo\images\train\2016_1.png")  # 加载图像，请根据您的图像路径进行更改

# 预处理图像
input_image = preprocess(image)
input_image = input_image.unsqueeze(0)  # 添加批次维度

# 使用模型进行推理
with torch.no_grad():
    output = model(input_image)

# 解析模型输出
# 这里根据您的模型输出格式进行解析，假设模型输出的是检测框的坐标、类别和置信度

# 示例：解析模型输出为检测框坐标、类别和置信度
boxes = output['boxes']  # 检测框坐标 (x1, y1, x2, y2)
labels = output['labels']  # 检测框类别
scores = output['scores']  # 检测框置信度

# 将检测结果保存为YOLO格式
with open('detections.txt', 'w') as f:
    for box, label, score in zip(boxes, labels, scores):
        # 转换检测框坐标为YOLO格式 (x_center, y_center, width, height)
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        width = box[2] - box[0]
        height = box[3] - box[1]

        # 将结果写入文件
        f.write(f"{label} {x_center} {y_center} {width} {height} {score}\n")

# 在原始图像上绘制检测框
draw = ImageDraw.Draw(image)
for box, label, score in zip(boxes, labels, scores):
    draw.rectangle([box[0], box[1], box[2], box[3]], outline="red")
    draw.text((box[0], box[1]), f"{label}: {score}", fill="red")

# 保存绘制了检测框的图像
image.save('detections.jpg')
