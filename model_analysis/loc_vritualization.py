import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体 SimHei为黑体
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

dataset_name = ["Restaurant", "Laptop", "Twitter"]
without_value = [80.14, 72.22, 70.47]
with_value = [80.42, 72.92, 71.72]

without_value_2 = [80.42, 73.95, 71.72]
with_value_2 = [81.25, 74.56, 72.81]

x = np.arange(len(dataset_name))  # the label locations


width = 0.20  # the width of the bars
fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, without_value_2, width, label='without')
rects2 = ax.bar(x + width/2, with_value_2, width, label='with')


# rects1 = ax.bar(x - width/2, without_value, width, label='without')
# rects2 = ax.bar(x + width/2, with_value, width, label='with')

# hehe = [e+0.5 for e in range(0, 91)]

# Add some text for labels, title and custom x-axis tick labels, etc.
# plt.yticks([e+5 for e in range(0, 100, 5)])

# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')          # 指定下边的边作为 x 轴   指定左边的边为 y 轴

# ax.spines['bottom'].set_position(('data', 65))
# ax.spines['left'].set_position(('data', 1))

plt.tick_params(labelsize=15)
ax.set_ylim(60, 85, 5)  # 设置y轴刻度的范围
ax.set_ylabel("准确率(%)", fontsize=18)
ax.set_xticks(x)  # 头顶的值
ax.set_xticklabels(dataset_name, fontsize=18)  # with、without
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            '{}'.format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 1),  # 2 points vertical offset
            textcoords="offset points",
            ha='center',
            va='bottom'
        )


autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.show()
