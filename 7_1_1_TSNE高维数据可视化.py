import numpy as np
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

'''
t-SNE高维数据可视化（python）

t-SNE（t-distributedstochastic neighbor embedding ）是目前最为流行的一种高维数据降维的算法。
在大数据的时代，数据不仅越来越大，而且也变得越来越复杂，数据维度的转化也在惊人的增加，
例如，一组图像的维度就是该图像的像素个数，其范围从数千到数百万。

对计算机而言，处理高维数据绝对没问题，但是人类能感知的确只有三个维度，
因此很有必要将高维数据可视化的展现出来。那么如何将数据集从一个任意维度的降维到二维或三维呢。
T-SNE就是一种数据降维的算法，其成立的前提是基于这样的假设：尽管现实世界中的许多数据集是嵌入在高维空间中，
但是都具有很低的内在维度。也就是说高维数据经过降维后，在低维状态下更能显示出其本质特性。
这就是流行学习的基本思想，也称为非线性降维。

关于t-SNE的详细介绍可以参考：
https://www.oreilly.com/learning/an-illustrated-introduction-to-the-t-sne-algorithm

下面就展示一下如何使用t-SNE算法可视化sklearn库中的手写字体数据集。
'''
# Random state.
RS = 20150101

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
digits = load_digits()
# We first reorder the data points according to the handwritten numbers.
X = np.vstack([digits.data[digits.target==i]
               for i in range(10)])
y = np.hstack([digits.target[digits.target==i]
               for i in range(10)])
digits_proj = TSNE(random_state=RS).fit_transform(X)

def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

scatter(digits_proj, y)
plt.savefig('digits_tsne-generated.png', dpi=120)
plt.show()