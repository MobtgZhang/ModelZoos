# 3 k近邻算法
## 3.1 k近邻算法
k邻近算法主要描述如下所示:给定一个数据集,对新的输入实例,在训练数据集中找到与该实例最邻近的k个实例,这k个实例的多数属于某个类,就将该输入实例分为这个类.

**k近邻算法**

输入:训练数据集
$$T=\left\{(x_{1},y_{1}),(x_{2},y_{2}),\cdots,(x_{N},y_{N})\right\}$$

其中,$x_{i}\in{\mathcal{X}\subseteq{\bm{R}^{n}}}$为实例的特征向量,$y_{i}\in\mathcal{Y}=\left\{y_{1},y_{2},\cdots,y_{K}\right\}$为实例类别,$i=1,2,\cdots,N$;实例特征向量$x$;

输出:实例$x$所属的类别$y$

(1) 根据给定的距离度量,在训练集$T$中找出与$x$最邻近的$k$个点,涵盖这$k$个点的$x$的邻域记作$N_{k}(x)$;

(2) 在$N_{k}(x)$中根据分类决策规则(例如有多数表决)决定$x$的类别$y$;

$$y=\arg\min\limits_{c_{j}}\sum\limits_{x_{i}\in{N_{k}(x)}}I\left(y_{i}=c_{j}\right)$$

上述表达式中,$I$为指示函数,即当$y_{i}=c_{j}$为$I=1$,否则$I=0$.

## 3.2 k近邻模型

$k$近邻算法使用的模型实际上对应于特征空间的划分.模型由三个基本要素:距离度量,$k$值的选择和分类决策规则决定.

**模型**：k近邻算法中,当训练集,距离度量(欧氏距离),$k$值以及分类决策规则(多数表决)确定后,对于任何一个新的输入实例,它所属类型则被唯一地确定.这相当于根据上树要求将特征空间划分为一些子空间,确定子空间中的每个点所属的类.

特征空间当中对于每个训练实例点$x_{i}$,距离该点比其他点更近的所有点组成一个区域,称之为单元.每个训练实例点拥有一个单元,所有训练实例点的单元构成


## 3.3 k近邻法的实现:kd树


