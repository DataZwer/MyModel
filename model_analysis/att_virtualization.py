import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

f, ax1 = plt.subplots(figsize=(6, 4), nrows=1)

# cmap用cubehelix map颜色
cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)

pt = np.random.rand(1, 10)

sns.heatmap(pt, linewidths=0.05, ax=ax1, vmax=1, vmin=0, cmap=cmap)
ax1.set_title('cubehelix map')
ax1.set_xticklabels([])  # 设置x轴图例为空值
ax1.set_ylabel('kind')
plt.show()
