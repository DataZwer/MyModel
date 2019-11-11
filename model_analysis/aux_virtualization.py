import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams['font.sans-serif']=['SimHei']  # 指定默认字体 SimHei为黑体
mpl.rcParams['axes.unicode_minus']=False  # 用来正常显示负号
acc_res = [79.04, 79.6, 80.14, 81.25]
acc_lap = [71.53, 73.43, 74.48, 74.65]
acc_twi = [70.46, 71.56, 72.5, 72.81]

plt.figure(figsize=(7, 5))

category = [0, 1, 2, 3]
line1, = plt.plot(category, acc_res, linewidth=3, marker='o', ms=9, label='Restaurant')
line2, = plt.plot(category, acc_lap, linewidth=3, marker='*', ms=9, label='Laptop')
line3, = plt.plot(category, acc_twi, linewidth=3, marker='^', ms=9, label='Twitter')
line1.set_dashes([5, 2])
line2.set_dashes([5, 2])
line3.set_dashes([5, 2])
plt.legend(fontsize=13)  # 让图例生效
plt.xticks([0, 1, 2, 3])

plt.tick_params(labelsize=15)
plt.xlabel("辅助特征数量", fontsize=18)  # X轴标签
plt.ylabel("准确率(%)", fontsize=18)  # Y轴标签
plt.show()

