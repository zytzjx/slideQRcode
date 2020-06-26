from matplotlib import pyplot as plt
import numpy as np
import cv2


img = cv2.imread('crop.png')
if img is None:
    print("图片读入失败")
    exit()

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# '''
# 	1Darray: 一维数组　这里通过gray.ravel()，把灰度图变为一维数组．
# 	bins: 统计分隔区间　如果是256 就是分成256份统计, 你可以修改这个值, 看不同的统计效果
# 	range: 统计数值的空间
# '''
# plt.hist(gray.ravel(), bins=256, range=[0, 256])
# plt.show()

# # Matplotlib预设的颜色字符
# bgrColor = ('b', 'g', 'r')


# for cidx, color in enumerate(bgrColor):
#     # cidx channel 序号
#     # color r / g / b
#     cHist = cv2.calcHist([img], [cidx], None, [256], [0, 256])
#     # 绘制折线图
#     plt.plot(cHist, color=color)  


# # 设定画布的范围
# plt.xlim([0, 256])

# # 显示画面
# plt.show()

# # 创建画布
# fig, ax = plt.subplots()

# # Matplotlib预设的颜色字符
# bgrColor = ('b', 'g', 'r')

# # 统计窗口间隔 , 设置小了锯齿状较为明显 最小为1 最好可以被256整除
# bin_win  = 4
# # 设定统计窗口bins的总数
# bin_num = int(256/bin_win)
# # 控制画布的窗口x坐标的稀疏程度. 最密集就设定xticks_win=1
# xticks_win = 2

# for cidx, color in enumerate(bgrColor):
#     # cidx channel 序号
#     # color r / g / b
#     cHist = cv2.calcHist([img], [cidx], None, [bin_num], [0, 256])
#     # 绘制折线图
#     ax.plot(cHist, color=color)  


# # 设定画布的范围
# ax.set_xlim([0, bin_num])
# # 设定x轴方向标注的位置
# ax.set_xticks(np.arange(0, bin_num, xticks_win))
# # 设定x轴方向标注的内容
# ax.set_xticklabels(list(range(0, 256, bin_win*xticks_win)),rotation=45)

# # 显示画面
# plt.show()

# 将图片转换为HSV格式
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# 创建画布
fig, ax = plt.subplots()
# Matplotlib预设的颜色字符
hsvColor = ('y', 'g', 'k')
# 统计窗口间隔 , 设置小了锯齿状较为明显 最小为1 最好可以被256整除
bin_win  = 3
# 设定统计窗口bins的总数
bin_num = int(256/bin_win)
# 控制画布的窗口x坐标的稀疏程度. 最密集就设定xticks_win=1
xticks_win = 2
# 设置标题
ax.set_title('HSV Color Space')
lines = []
for cidx, color in enumerate(hsvColor):
    # cidx channel 序号
    # color r / g / b
    cHist = cv2.calcHist([img], [cidx], None, [bin_num], [0, 256])
    # 绘制折线图
    line, = ax.plot(cHist, color=color,linewidth=1)
    lines.append(line)  

# 标签
labels = [cname +' Channel' for cname in 'HSV']
# 添加channel 
plt.legend(lines,labels, loc='upper right')
# 设定画布的范围
ax.set_xlim([0, bin_num])
# 设定x轴方向标注的位置
ax.set_xticks(np.arange(0, bin_num, xticks_win))
# 设定x轴方向标注的内容
ax.set_xticklabels(list(range(0, 256, bin_win*xticks_win)),rotation=45)

# 显示画面
plt.show()