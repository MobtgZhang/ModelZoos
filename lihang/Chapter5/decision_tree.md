# 决策树
## 1.决策树模型与学习
### 1.1 **定义(决策树)**

分类决策树是一种描述对实例进行分类的树形结构.决策树由结点和有向边组成.结点有两种类型:内部结点和叶结点.内部结点表示一个特征或者属性,叶结点表示一个类.

### 1.2 **决策树学习**

假定给定训练数据集
$$D=\left\{(x_{1},y_{1}),(x_{2},y_{2}),\cdots,(x_{N},y_{N})\right\}$$
其中,$x_{i}=\left\{x_{i}^{(1)},x_{i}^{(2)},\cdots,x_{i}^{(n)}\right\}^{T}$表示的输入实例(特征向量),$n$为特征个数,$y_{i}\in{1,2,\cdots,K}$为类的标记,$i=\left\{1,2,\dots,K\right\}$表示类标记,$i=1,2,\cdots,N$,$N$表示的是样本的容量.决策树学习的目标就是根据给定的训练数据集构建一个决策树模型,使得它能够对实例进行正确的分类.
## 2. 特征选择
### 2.1 信息增益

在信息论与概率统计当中,熵是表示随机变量不确定性的度量.设$X$是一个取有限个值的离散随机变量,其中概率分布如下所示:
$$P\left(X=x_{i}\right)=p_{i},i=1,2,\cdots,n$$
那么随机变量$X$的熵定义为
$$H(X)=-\sum\limits_{i=1}^{n}p_{i}\log{p_{i}}$$
由定义可知,熵只依赖于$X$的分布,而与$X$的取值是无关的,所以可将$X$的熵记作$H(p)$,即如下的形式
$$H(p)=-\sum\limits_{i=1}^{n}p_{i}\log{p_{i}}$$
熵越大,随机变量的不确定就会越大,从定义当中就可以得到
$$0\leq{H(p)}\leq{\log{n}}$$

假设随机变量$(X,Y)$,其中联合概率分布为
$$P\left(X=x_{i},Y=y_{j}\right)=p_{ij},i=1,2,\dots,n;j=1,2,\cdots,m$$
条件熵$H\left(Y\left|X\right.\right)$表示已知随机变量$X$的条件下随机变量$Y$的不确定性.随机变量$X$给定的条件下随机变量$Y$的条件熵$H\left(Y\left|X\right.\right)$,定义为$X$给定条件下$Y$的条件概率分布的熵对$X$的数学期望
$$H\left(Y\left|X\right.\right)=\sum\limits_{i=1}^{n}p_{i}H\left(Y\left|X=x_{i}\right.\right)$$
其中$p_{i}=P\left(X=x_{i}\right),i=1,2,\cdots,n$.

当熵和条件熵中的概率由数据估计(特别是极大似然估计)得到时候,所对应的熵与条件熵分别称为经验熵和经验条件熵.

信息增益表示得知特征$X$的信息而使得类$Y$的信息的不确定性减少的程度.
**定义(信息增益)**：特征$A$对训练数据集$D$的信息增益$g(D,A)$,定义为集合$D$的经验熵$H(D)$与特征$A$给定条件下$D$的经验条件熵$H\left(D\left|A\right.\right)$之差,即$$g(D,A)=H(D)-H\left(D\left|A\right.\right)$$
一般地,熵$H(Y)$与条件熵$H\left(Y\left|X\right.\right)$之差称为互信息.决策树学习中的信息增益等价于训练数据集中类与特征的互信息.

根据信息增益准则的特征选择方法是：对训练数据集（或者子集）$D$,计算其中每个特征的信息增益,并比较它们的大小,选择信息增益最大的特征.

设训练数据集为$D$,$\left|D\right|$表示的是其中样本的容量,即样本个数.假设有$K$个类$C_{k},k=1,2,\cdots,K$,$\left|C_{k}\right|$表示的是类$C_{k}$的样本个数,并且有$\sum\limits_{k=1}^{K}\left|C_{k}\right|=\left|D\right|$.假设特征$A$有$n$个不同的取值$\left\{a_{1},a_{2},\cdots,a_{n}\right\}$,根据特征$A$的取值将$D$划分为$n$个子集$D_{1},D_{2},\cdots,D_{n}$,其中$\left|D_{i}\right|$表示的是集合$D_{i}$的样本个数,$\sum\limits_{i=1}^{n}\left|D_{i}\right|=\left|D\right|$.记子集$D_{i}$中属于类$C_{k}$的样本集合为$D_{ik}$,即$D_{ik}=D_{i}\cap{C_{k}}$,$\left|D_{ik}\right|$为$D_{ik}$的样本个数,信息增益的算法如下所示

**信息增益的算法**

**输入**：训练数据集$D$和特征$A$

**输出**：特征$A$对训练数据集$D$的信息增益$g(D,A)$

(1) 计算数据集$D$的经验熵$H(D)$:
$$H(D)=-\sum\limits_{k=1}^{K}\dfrac{\left|C_{k}\right|}{\left|D\right|}\log_{2}\dfrac{\left|C_{k}\right|}{\left|D\right|}$$

(2) 计算特征$A$对数据集$D$的经验条件熵$H\left(D\left|A\right.\right)$
$$H\left(D\left|A\right.\right)=\sum\limits_{i=1}^{n}\dfrac{\left|D_{i}\right|}{\left|D\right|}H\left(D_{i}\right)=\sum\limits_{i=1}^{n}\dfrac{\left|D_{i}\right|}{\left|D\right|}\sum\limits_{k=1}^{K}\dfrac{\left|D_{ik}\right|}{\left|D_{i}\right|}\log_{2}\dfrac{\left|D_{ik}\right|}{\left|D_{i}\right|}$$

(3) 计算信息增益

$$g(D,A)=H\left(D\right)-H\left(D\left|A\right.\right)$$

### 2.2 信息增益比
以信息增益作为划分训练数据集的特征,存在偏向于选择取值较多的特征问题,使用信息增益比可以对这一问题进行校正.

**定义(信息增益比)**：特征$A$对训练数据集$D$的信息增益比$g_{R}(D,A)$定义为其信息增益$g(D,A)$与训练数据集$D$关于特征$A$的值的熵$H_{A}(D)$之比,即
$$g_{R}(D,A)=\dfrac{g(D,A)}{H_{A}(D)}$$

其中,$H_{A}(D)=-\sum\limits_{i=1}^{n}\dfrac{\left|D_{i}\right|}{\left|D\right|}\log_{2}\dfrac{\left|D_{i}\right|}{\left|D\right|}$,其中$n$是特征$A$取值的个数.

## 3. 决策树的生成
ID3算法的核心是在决策树各个结点上应用信息增益准则选择特征,递归地构建决策树.具体方法是:从根节点开始,对结点计算所有可能特征的信息增益,选择信息增益最大的特征作为结点的特征,由该特征的不同取值建立子结点:再对子结点递归地调用以上方法,构建决策树;直到所有特征的信息增益均很小或者没有特征可以选择为止,最后得到一棵决策树,ID3相当于使用极大似然的方法进行概率模型的选择.

### 3.1 ID3算法

**输入**:训练数据集$D$,特征集$A$阈值$\epsilon$;

**输出**:决策树$T$;

(1) 若$D$中所有实例属于同一类$C_{k}$,则$T$为单结点树,并且将类$C_{k}$作为该结点的类标记,返回$T$;

(2) 若$A=\empty$,则$T$为单结点树,并将$D$中实例数最大的类$C_{k}$作为该结点的类标记,返回$T$;

(3) 否则,按照上述信息增益的算法计算$A$中各特征对$D$的信息增益,选择信息增益最大的特征$A_{g}$;

(4) 如果$A_{g}$的信息增益小于阈值$\epsilon$,则置$T$为单结点树,并且将$D$中实例数最大的类$C_{k}$作为该结点的类标记,返回$T$;

(5) 否则,对$A_{g}$的每一可能值$a_{i}$,依$A_{g}=a_{i}$将$D$分割为若干非空子集$D_{i}$,将$D_{i}$中实例数最大的类作为标记,构建子结点,由结点及子结点构成树$T$,返回$T$;

(6) 对第$i$个子结点,以$D_{i}$为训练集,以$A-\left\{A_{g}\right\}$为特征集,递归地调用(1)-(5)步骤,得到子树$T_{i}$,返回$T_{i}$.

### 3.2 C4.5的生成算法
C4.5算法与ID3算法是相似的,C4.5算法对ID3算法进行了改进.C4.5在生成的过程中,用信息增益之比来选择特征.

**输入**：训练数据集$D$,特征集$A$阈值$\epsilon$;

**输出**：决策树$T$.

(1) 如果$D$中所有实例均属于同一类$C_{k}$,则置$T$为单结点树,并将$C_{k}$作为该结点的类,返回$T$;

(2) 如果$A=\empty$,则置$T$为单结点树,并将$D当中实例最大的类$C_{k}$作为该结点的类,返回$T$;

(3) 否则,按照信息增益之比计算$A$中各特征对$D$的信息增益比,选择信息增益之比最大的特征$A_{g}$;

(4) 如果$A_{g}$信息增益比小于阈值$\epsilon$,则将$T$置于单结点树,并将$D$当中实例数最大的类$C_{k}$作为该结点的类,返回$T$;

(5) 否则,对$A_{g}$的每一可能值$a_{i}$,依照$A_{g}=a_{i}$将$D$分割为子集若干非空$D_{i}$,将$D_{i}$当中实例数最大的类作为标记,构建子结点,由子结点及其子结点构成树$T$,返回$T$;

(6) 对结点$i$,以$D_{i}$为训练集,以$A-\left\{A_{g}\right\}$为特征集,递归地调用步骤(1)-(5),得到子树$T_{i}$,返回树$T_{i}$.

## 4. 决策树的剪枝

决策树生成算法递归地产生决策树,直到不能继续下去为止.这样产生的树往往对训练数据的分类很准确,但是对于未知的测试数据分类却没有那么准确,即出现过拟合现象.过拟合的原因在于学习的时候过多地考虑如何提高对训练数据的正确分类,从而构建出过于复杂的决策树,解决这个问题的办法是考虑决策树的复杂度,对已经生成的决策树进行简化.

在决策树学习中将已经生成的树进行简化的过程称为剪枝,具体地,剪枝从已经生成的树上裁剪掉一些子树或者叶结点，并且将其根节点或者父结点作为新的叶结点,从而简化分类树模型.

其中的一种决策树剪枝算法
决策树的剪枝往往通过极小化决策树整体的损失函数或者是代价函数来实现.设树$T$的叶结点个数为$\left|T\right|$,$t$是树$T$的叶结点,该叶结点有$N_{t}$个样本点,其中$k$类的样本点有$N_{tk}$个,$k=1,2,\cdots,K$,$H_{t}(T)$为叶结点$t$上经验熵,$\alpha\geq{0}$为参数,则决策树学习的损失函数可以定义为以下的形式:

$$C_{\alpha}\left(T\right)=\sum\limits_{t=1}^{\left|T\right|}N_{t}H_{t}\left(T\right)+\alpha\left|T\right|$$

其中经验熵表示为如下的形式

$$H_{t}(T)=-\sum\limits_{k}\dfrac{N_{tk}}{N_{t}}\log\dfrac{N_{tk}}{N_{t}}$$

在损失函数中,记

$$C(T)=\sum\limits_{t=1}^{\left|T\right|}N_{t}H_{t}(T)=-\sum\limits_{t=1}^{\left|T\right|}\sum\limits_{k=1}^{K}\dfrac{N_{tk}}{N_{t}}\log\dfrac{N_{tk}}{N_{t}}$$

所以这样就会有以下的形式

$$C_{\alpha}(T)=C(T)+\alpha\left|T\right|$$

上述表达式中,$C(T)$表示模型对训练数据的预测误差,即模型与训练数据的拟合程度,$T$表示模型的复杂度,参数$\alpha\geq{0}$控制两者之间的影响.较大的$\alpha$促使选择较为简单的模型(树),较小的$\alpha$促使模型选择较为复杂的树.$\alpha=0$意味着仅仅考虑模型与训练数据的拟合程度,不考虑模型的复杂度.

所以剪枝就是当$\alpha$确定的时候,选择损失函数最小的模型,即损失函数最小的子树.当$\alpha$值确定的时候,子树越大,往往与训练数据的拟合越好,但是模型的复杂度就会越高;相反,子树越小的时候,模型的复杂度就会越低,但是往往与训练数据的拟合程度并不好,损失函数正恰当表示了对两者之间的平衡.

可以看出决策树生成只考虑了通过提高信息增益(或者是信息增益之比)对训练数据进行更好的拟合.而决策树剪枝通过优化损失函数,同时还考虑到了减小模型的复杂度.决策树生成学习局部的模型,而决策树剪学习整体的模型.

**决策树剪枝算法**

**输入**：生成算法产生的整个树$T$,参数$\alpha$;
**输出**：修建之后的子树$T_{\alpha}$.

(1) 计算每个结点的经验熵;
(2) 递归地从树的叶结点向上回缩.设一组叶结点回缩到其父结点之前与之后的整体树分别为$T_{B}$与$T_{A}$,其对应的损失函数值分别是$C_{\alpha}(T_{B}),C_{\alpha}(T_{A})$,若

$$C_{\alpha}(T_{A})\leq{C_{\alpha}(T_{B})}$$

则进行剪枝,即将父结点变为新的叶结点.

(3) 返回(2),直到不能继续为止,得到损失函数最小的子树$T$.

## 5. CART算法
CART是在给定输入随机变量$X$条件下输出随机变量$Y$的条件概率分布的学习方法.CART假设决策树是二叉树,内部结点特征的取值为"是"和"否";向左分支是取值为"是"的分支,向右分支是取值为"否"的分支.这样的决策树等价于递归地二分每个特征,将输入空间即特征空间划分为有限个单元,并且在这些单元上确定预测的概率分布,也就是在输入给定的条件下输出的条件概率分布.

CART算法是由以下的两步组成算法:
(1) 决策树生成:基于训练数据集生成决策树,生成的决策树要尽量大;
(2) 决策树剪枝:用验证数据集对已经生成的树进行剪枝并且选择最优的子树,这时候用损失函数最小作为剪枝的标准.

### 5.1 CART生成

决策树的生成就是递归地构建二叉树的过程,对回归树使用平方误差最小化准则,对分类树使用基尼指数最小化准则,进行特征选择,生成二叉树.

一般地分为两种问题,一种是回归问题,另外一种是分类问题.

1. 回归树的生成

假设$X$与$Y$分别为输入和输出变量,并且$Y$是连续变量,给定数据集

$$D=\left\{(x_{1},y_{1}),(x_{2},y_{2}),\cdots,(x_{N},y_{N})\right\}$$

这样考虑如何生成回归树.

一颗回归树对应着输入空间(特征空间)的一个划分以及在划分的单元上的输出值.假设已将输入空间划分为$M$个单元$R_{1},R_{2},\cdots,R_{M}$,并且在每个单元$R_{m}$上有一个固定的输出值$c_{m}$,于是回归模型可以表示为以下的形式

$$f(x)=\sum\limits_{m=1}^{M}c_{m}I\left(x\in{R_{m}}\right)$$

当输入空间的划分确定的时候,可以使用平方误差$\sum\limits_{x_{i}\in{R_{m}}}\left(y_{i}-f(x_{i})\right)^{2}$来表示回归树对于训练数据的预测误差,用平方误差最小的准则求解每个单元上的最优输出值.易知,单元$R_{m}$上的$c_{m}$的最优值$\hat{c}_{m}$是$R_{m}$上所有输入实例$x_{i}$对应$y_{i}$的均值,即

$$\hat{c}_{m}=ave\left(y_{i}\left|x_{i}\in{R_{m}}\right.\right)$$

这里使用到的是启发式的方法,选择第$j$个变量$x^{(j)}$和它的值$s$,作为切分变量和切分点,并且定义两个区域

$$R_{1}(j,s)=\left\{x\left|x^{(j)}\leq{s}\right.\right\},R_{2}(j,s)=\left\{x\left|x^{(j)}>{s}\right.\right\}$$

然后寻找最优切分变量$j$和最优切分点$s$,即求解

$$\min\limits_{j,s}\left[\min\limits_{c_{1}}\sum\limits_{x_{i}\in{R_{1}(j,s)}}\left(y_{i}-c_{1}\right)^{2}+\min\limits_{c_{2}}\sum\limits_{x_{i}\in{R_{2}(j,s)}}\left(y_{i}-c_{2}\right)^{2}\right]$$

对固定输入变量$j$可以找到最优切分点$s$.

$$\hat{c}_{1}=ave\left(y_{i}\left|x_{i}\in{R_{1}(j,s)}\right.\right),\hat{c}_{2}=ave\left(y_{i}\left|x_{i}\in{R_{2}(j,s)}\right.\right)$$

遍历所有输入变量,找到最优的切分变量$j$,构成一个对$(j,s)$.一次将输入空间划分为两个区域,接着对每个区域重复上述划分过程,直到满足停止条件为止.这样就会生成一颗回归树.这样回归树通常称为最小回归树.




2. 分类树的生成
