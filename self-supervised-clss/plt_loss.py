import matplotlib.pyplot as plt

filename = r'D:\efficientteacher-main\self-supervised-smiclr\stage1_loss.txt'
x = []
i = 0
with open(filename, 'r') as file:
    data = file.readline().split()  # 假设数据用空格分隔，如果是逗号请使用split(',')
    data = list(map(float, data))  # 将字符串数据转换为浮点数

for a in range(len(data)):
    i = i + 1
    x.append(i)

# 创建折线图
plt.plot(x, data)  # marker='o' 表示标记每个点
plt.title('Fold Line Graph')  # 图表标题
plt.xlabel('X Axis')  # X轴标签
plt.ylabel('Y Axis')  # Y轴标签

# 显示图表
plt.show()
