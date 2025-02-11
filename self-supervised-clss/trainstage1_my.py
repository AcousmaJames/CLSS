# trainstage1.py
import torch, argparse, os
import yaml

import net_my, config, loaddataset,loaddataset_my,net
import DictToClass
# train stage one
from configs.defaults import get_cfg


def train(args):
    if torch.cuda.is_available() and config.use_gpu:
        # DEVICE = torch.device("cuda:" + str(config.gpu_name))
        DEVICE = torch.device("cuda:" + str(0))
        # 每次训练计算图改动较小使用，在开始前选取较优的基础算法（比如选择一种当前高效的卷积算法）
        # torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")
    print("current deveice:", DEVICE)

    # train_dataset = loaddataset.PreDataset(root='dataset', train=True, transform=config.train_transform, download=True)
    train_dataset = loaddataset_my.PreDataset(images_dir=args.images_dir, transform=config.train_transform)
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16,
                                             drop_last=True)

    # f = open(args.c)
    # y = yaml.safe_load(f)
    cfg = DictToClass.ReadConfigFiles.cfg(args.c)
    # model = net_my.SimCLRStage1(cfg).to(DEVICE)
    # lossLR = net_my.Loss().to(DEVICE)
    model = net.SimCLRStage1().to(DEVICE)
    lossLR = net.Loss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    os.makedirs(config.save_path, exist_ok=True)
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        total_loss = 0
        for batch, (imgL, imgR) in enumerate(train_data):
            imgL, imgR = imgL.to(DEVICE), imgR.to(DEVICE)

            _, pre_L = model(imgL)
            _, pre_R = model(imgR)

            loss = lossLR(pre_L, pre_R, args.batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("epoch", epoch, "batch", batch, "loss:", loss.detach().item())
            total_loss += loss.detach().item()

        print("epoch loss:", total_loss / len(train_dataset) * args.batch_size)

        with open(os.path.join(config.save_path, "stage1_loss.txt"), "a") as f:
            f.write(str(total_loss / len(train_dataset) * args.batch_size) + " ")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(config.save_path, 'model_stage1_epoch' + str(epoch) + '.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--c', default=r'D:\efficientteacher-main\configs\sup\custom\yolov5l_custom.yaml')
    parser.add_argument('--batch_size', default=1, type=int, help='')
    parser.add_argument('--max_epoch', default=1000, type=int, help='')
    parser.add_argument('--images_dir',
                        default=r'D:\efficientteacher-main\self-supervised-smiclr\dataset\image-test',
                        help="all images for self-super")
    args = parser.parse_args()
    train(args)
