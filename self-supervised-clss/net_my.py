# net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
from models.backbone.yolov5rfa_backbone import YoloV5RfaBackBone
from models.neck.yolov5rfa_neck import YoloV5RfaNeck

# stage one ,unsupervised learning



class SimCLRStage1(nn.Module):
    def __init__(self,  feature_dim=128):
        super(SimCLRStage1, self).__init__()

        # 使用自定义的YoloV5BackBone
        self.f = YoloV5RfaBackBone()
        self.neck = YoloV5RfaNeck()
        # print(f)
        # 获取YoloV5BackBone的输出维度，假设是sppf的输出维度
        # backbone_output_dim = self.neck.channels_out['spp']
        backbone_output_dim = self.neck.output_p5

        # projection head
        self.g = nn.Sequential(
            nn.Linear(backbone_output_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True)
        )

    def forward(self, x):
        # 使用自定义的backbone提取特征
        x1 = self.f(x)
        x2, x3, h = self.neck(x1)
        # 使用projection head生成最终的特征
        z = self.g(h.view(h.size(0), -1))
        return F.normalize(h, dim=-1), F.normalize(z, dim=-1)


# stage two ,supervised learning
class SimCLRStage2(torch.nn.Module):
    def __init__(self, num_class):
        super(SimCLRStage2, self).__init__()
        # encoder

        self.f = SimCLRStage1().f
        backbone_output_dim = self.f.channels_out['spp']
        # classifier
        self.fc = nn.Linear(backbone_output_dim, num_class, bias=True)

        for param in self.f.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss,self).__init__()

    def forward(self,out_1,out_2,batch_size,temperature=0.5):
        # 分母 ：X.X.T，再去掉对角线值，分析结果一行，可以看成它与除了这行外的其他行都进行了点积运算（包括out_1和out_2）,
        # 而每一行为一个batch的一个取值，即一个输入图像的特征表示，
        # 因此，X.X.T，再去掉对角线值表示，每个输入图像的特征与其所有输出特征（包括out_1和out_2）的点积，用点积来衡量相似性
        # 加上exp操作，该操作实际计算了分母
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # 分子： *为对应位置相乘，也是点积
        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        return (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()


if __name__=="__main__":
    model = SimCLRStage1()
    for name, module in SimCLRStage1().named_children():
        print(name,module)

