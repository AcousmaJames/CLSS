import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from yolov5_rfa_neck import SimCLRStage1

COLORS = np.random.uniform(0, 255, size=(80, 3))

# 解析YOLOv5的检测结果
def parse_detections(results):
    detections = results.pandas().xyxy[0]
    detections = detections.to_dict()
    boxes, colors, names = [], [], []

    for i in range(len(detections["xmin"])):
        confidence = detections["confidence"][i]
        if confidence < 0.2:
            continue
        xmin = int(detections["xmin"][i])
        ymin = int(detections["ymin"][i])
        xmax = int(detections["xmax"][i])
        ymax = int(detections["ymax"][i])
        name = detections["name"][i]
        category = int(detections["class"][i])
        color = COLORS[category]

        boxes.append((xmin, ymin, xmax, ymax))
        colors.append(color)
        names.append(name)
    return boxes, colors, names

# 将检测结果画在图片上
def draw_detections(boxes, colors, names, img):
    for box, color, name in zip(boxes, colors, names):
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(img, name, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, lineType=cv2.LINE_AA)
    return img


# 对检测框内部的特征进行归一化，将检测框外部的特征设置为0
def renormalize_cam_in_bounding_boxes(boxes, colors, names, image_float_np, grayscale_cam):
    """Normalize the CAM to be in the range [0, 1] 
    inside every bounding boxes, and zero outside of the bounding boxes. """
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())    
    renormalized_cam = scale_cam_image(renormalized_cam)
    eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
    image_with_bounding_boxes = draw_detections(boxes, colors, names, eigencam_image_renormalized)
    return image_with_bounding_boxes


# 1.读图片，做预处理
image_url = r"G:\wkkSet\henanSet\images\2016_13.png"
img = cv2.imread(image_url)[:,:,[2,1,0]]
img = cv2.resize(img, (1024, 1024))
rgb_img = img.copy()     # rgb_img作为网络输入端的图片
img = np.float32(img) / 255    # 使用show_cam_on_image进行可视化时使用
transform = transforms.ToTensor()
tensor = transform(img).unsqueeze(0)    # 在获取特征图时作为输入数据使用

# 2.模型初始化
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model = SimCLRStage1()
state_dict = torch.load(r"D:\efficientteacher-main\runs\175-220-1024\rfa-self3900-350=dan\exp2\weights\best.pt")
model.load_state_dict(state_dict,strict=False)
model.eval()
model.cpu()

# 3.模型推理、推理结果可视化
results = model([rgb_img])
boxes, colors, names = parse_detections(results)
detections = draw_detections(boxes, colors, names, rgb_img.copy())
cv2.imwrite('./detection.png', detections[:,:,[2,1,0]])

# 4.指定要可视化的特征的层，这里指定detect的前一个层
target_layers = [model.model.model.model[-2]]

# 5.实例化EigenCAM、得到可视化特征，并显示在原图上
cam = EigenCAM(model, target_layers,use_cuda=False)
grayscale_cam = cam(tensor)[0, :, :]
cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
cv2.imwrite('./cam_image.png', cam_image[:,:,[2,1,0]])

# 6.对检测框内部的特征进行归一化，将检测框外部的特征设置为0
renormalized_cam_image = renormalize_cam_in_bounding_boxes(boxes, colors, names, img, grayscale_cam)
cv2.imwrite('./renormalized_cam_image.png', renormalized_cam_image[:,:,[2,1,0]])
cv2.imwrite('./all.png', np.hstack((rgb_img, cam_image, renormalized_cam_image))[:,:,[2,1,0]])