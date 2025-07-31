import matplotlib.pyplot as plt

# 读取txt文件
with open('./LR/0.9_learning_rates.txt', 'r') as file:
    data = file.readlines()

# 将数据转换为列表
data = [float(line.strip()) for line in data]

# 绘制折线图
plt.plot(data)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Data Visualization')
# plt.savefig('output.png')  # 保存为图片文件
plt.show()
