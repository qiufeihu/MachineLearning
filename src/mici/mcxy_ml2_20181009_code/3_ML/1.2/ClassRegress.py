# 使用逻辑回归进行分类的例子
# In[]
# 显示当前路径
import os
os.getcwd()

# In[]
import numpy as np
import pandas as pd
# 读入数据，为dataframe格式
df = pd.read_csv('iris.data', header=None)
# 显示前面行
df.head()

# In[]
# 显示后面行
df.tail()

# In[]
# 取dataframe中的数据到数组array
y = df.iloc[:, 4].values 
# 长度和类别
len(y), np.unique(y)

# In[]
# 把类别转为整数
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_i = le.fit_transform(y)
np.unique(y_i)

# In[]
# 前50个数据和最后50个数据
y_i[:50], y_i[50:100], y_i[-50:]

# In[]
# 获取两列特征数据
x = df.iloc[:, [2, 3]].values
# shape和前10行
x.shape, x[:10]

# In[]
# 分割训练集和测试集
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y_i, test_size=0.3, random_state=1, stratify=y)
len(y_train),len(y_test)

# In[]
# 统计每一种类型数量
for t in np.unique(y_train):
    print(t)
    print(list(y_train).count(t))

# In[]
for t in np.unique(y_test):
    print(t)
    print(list(y_test).count(t))

# In[]
# 数据标准化
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

x_train_std[:10]

# In[]
# 使用逻辑回归进行分类
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=100.0, random_state=1)
# 训练
lr.fit(x_train_std, y_train)

# In[]
# 载入画图函数
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')
    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='test set')
# In[]
# 画图
x_combined_std = np.vstack((x_train_std, x_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(x_combined_std, y_combined,
                      classifier=lr, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

#################################################
# 线性回归演示
# In[]
from IPython.display import Image

df = pd.read_csv('housing.data.txt',
                 header=None,
                 sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()

# In[]
df.tail()
# In[]
# 房间数与价格
x = df[['RM']].values
y = df['MEDV'].values

x.shape, y.shape, y[:, np.newaxis].shape
# In[]
# 数据标准化
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
sc_y = StandardScaler()

x_std = sc_x.fit_transform(x)
# 转为二维数据进行标准化，然后调用flatten再转为一维
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

# In[]
y_std.shape

# In[]:
# 线性模型训练
from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(x_std, y_std)
print("训练结束")

# In[]:
# 载入回归画图处理
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)    
    return 

# In[]:
y_pred = slr.predict(x)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)

# In[]:
lin_regplot(x, y, slr)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.show()


