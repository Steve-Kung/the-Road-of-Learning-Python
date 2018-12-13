
# 1 NumPy
1.NumPy系统是Python的一种开源的数值计算扩展，是一个用python实现的科学计算包.

2.NumPy（Numeric Python）提供了许多高级的数值编程工具，如：矩阵数据类型、矢量处理，以及精密的运算库。专为进行严格的数字处理而产生。

3.使用Anaconda发行版的Python，已经帮我们事先安装好了Numpy模块，因此无需另外安装

4.依照标准的Numpy约定，习惯使用 import numpy as np方式导入该模块

## 1.1 Numpy的数据结构：ndarry，一种多维数组对象
### 1.1.1 ndarray介绍
ndarray：N-dimensional array， N维数组
一种由相同类型的元素组成的多维数组

元素数量是事先指定好的

元素的数据类型由dtype（data-type）对象来指定，每个ndarray只有一种dtype类型

大小固定，创建好数组时一旦指定好大小，就不会再发生改变

ndim 维度数量

shape是一个表示各维度大小的元组，即数组的形状

dtype，一个用于说明数组元素数据类型的对象

size，元素总个数，即shape中各数组相乘


```python
import numpy as np
a = np.array([[[3.4, 5, 6, 8], [3, 2.4, 5, 7]], [[2.3, 4, 5, 6], [0.9, 5, 6, 1]], [[9, 6.7, 3, 2], [1, 3, 4, 5]]])
```


```python
a.ndim
```




    3




```python
a.dtype
```




    dtype('float64')




```python
a.shape
```




    (3, 2, 4)




```python
a.size
```




    24




```python
a[0]
```




    array([[3.4, 5. , 6. , 8. ],
           [3. , 2.4, 5. , 7. ]])




```python
a[1]
```




    array([[2.3, 4. , 5. , 6. ],
           [0.9, 5. , 6. , 1. ]])




```python
a[2]
```




    array([[9. , 6.7, 3. , 2. ],
           [1. , 3. , 4. , 5. ]])




```python
a[2][0]
```




    array([9. , 6.7, 3. , 2. ])




```python
a[2][0][3]
```




    2.0



### 1.1.2 ndarray的常见创建方式
()与[]成对使用即可

array函数：接收一个普通的Python序列，转成ndarray

zeros函数：创建指定长度或形状的全零数组

ones函数：创建指定长度或形状的全1数组

empty函数：创建一个没有任何具体值的数组（准确地说是一些未初始化的垃圾值）



```python
np.zeros((3, 4))
```




    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])




```python
np.ones((4, 6))
```




    array([[1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1.]])




```python
np.empty((2, 3, 4))
```




    array([[[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]],
    
           [[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]]])



### 1.1.3 ndarray的其他创建方式
arrange函数：类似于python的range函数，通过指定开始值、终值和步长来创建一维数组，注意数组不包括终值

linspace函数：通过指定开始值、终值和元素个数来创建一维数组，可以通过endpoint关键字指定是否包括终值，缺省设置是包括终值

logspace函数：和linspace类似，不过它创建等比数列

使用随机数填充数组，即使用numpy.random模块的random()函数，数组所包含的的元素数量由参数决定


```python
np.arange(20)
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
           17, 18, 19])




```python
np.arange(0, 20, 1)
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
           17, 18, 19])




```python
np.arange(0, 20, 2)
```




    array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18])



reshape()函数可以改变
数组的形状，
但是注意
元素总个数不能改变


```python
np.arange(0, 12).reshape(3, 4)
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])




```python
np.linspace(0, 10, 5)
```




    array([ 0. ,  2.5,  5. ,  7.5, 10. ])




```python
help(np.linspace)
```

    Help on function linspace in module numpy.core.function_base:
    
    linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
        Return evenly spaced numbers over a specified interval.
        
        Returns `num` evenly spaced samples, calculated over the
        interval [`start`, `stop`].
        
        The endpoint of the interval can optionally be excluded.
        
        Parameters
        ----------
        start : scalar
            The starting value of the sequence.
        stop : scalar
            The end value of the sequence, unless `endpoint` is set to False.
            In that case, the sequence consists of all but the last of ``num + 1``
            evenly spaced samples, so that `stop` is excluded.  Note that the step
            size changes when `endpoint` is False.
        num : int, optional
            Number of samples to generate. Default is 50. Must be non-negative.
        endpoint : bool, optional
            If True, `stop` is the last sample. Otherwise, it is not included.
            Default is True.
        retstep : bool, optional
            If True, return (`samples`, `step`), where `step` is the spacing
            between samples.
        dtype : dtype, optional
            The type of the output array.  If `dtype` is not given, infer the data
            type from the other input arguments.
        
            .. versionadded:: 1.9.0
        
        Returns
        -------
        samples : ndarray
            There are `num` equally spaced samples in the closed interval
            ``[start, stop]`` or the half-open interval ``[start, stop)``
            (depending on whether `endpoint` is True or False).
        step : float, optional
            Only returned if `retstep` is True
        
            Size of spacing between samples.
        
        
        See Also
        --------
        arange : Similar to `linspace`, but uses a step size (instead of the
                 number of samples).
        logspace : Samples uniformly distributed in log space.
        
        Examples
        --------
        >>> np.linspace(2.0, 3.0, num=5)
        array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])
        >>> np.linspace(2.0, 3.0, num=5, endpoint=False)
        array([ 2. ,  2.2,  2.4,  2.6,  2.8])
        >>> np.linspace(2.0, 3.0, num=5, retstep=True)
        (array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ]), 0.25)
        
        Graphical illustration:
        
        >>> import matplotlib.pyplot as plt
        >>> N = 8
        >>> y = np.zeros(N)
        >>> x1 = np.linspace(0, 10, N, endpoint=True)
        >>> x2 = np.linspace(0, 10, N, endpoint=False)
        >>> plt.plot(x1, y, 'o')
        [<matplotlib.lines.Line2D object at 0x...>]
        >>> plt.plot(x2, y + 0.5, 'o')
        [<matplotlib.lines.Line2D object at 0x...>]
        >>> plt.ylim([-0.5, 1])
        (-0.5, 1)
        >>> plt.show()
    
    


```python
np.logspace(0, 2, 5)
```




    array([  1.        ,   3.16227766,  10.        ,  31.6227766 ,
           100.        ])



0，表示10的零次方，2表示10的2次方，5表示生成的数组元素总数


```python
np.random.random((2, 2, 4))
```




    array([[[0.61640195, 0.72848392, 0.7337928 , 0.96618842],
            [0.56405401, 0.77861004, 0.19452196, 0.70480214]],
    
           [[0.66790088, 0.08849339, 0.6226763 , 0.60304195],
            [0.74353626, 0.44194598, 0.66915838, 0.34384812]]])



### 1.1.4 NumPy中的数据类型
创建NumPy数组时可以通过dtype属性显式指定数据类型，如果不指定，NumPy会自己推断出合适的数据类型，所以一般无需显式指定

astype方法，可以转换数组的元素数据类型，得到一个新数组


```python
ndarray02 = np.array([1, 2, 3 , 4])
ndarray02.dtype
```




    dtype('int32')




```python
ndarray03 = ndarray02.astype(float)
ndarray03.dtype
```




    dtype('float64')



数值型dtype的命名方式：一个类型名（比如int、float），后面接着一个用于表示各元素位长的数字

比如表中的双精度浮点值，即Python中的float对象，需要占用8个字节（即64位），因此该类型在NumPy中就记为float64


```python
d = np.array(["Python", "Scala", "Java", "C#"])
d
```




    array(['Python', 'Scala', 'Java', 'C#'], dtype='<U6')




```python
d.dtype
```




    dtype('<U6')




```python
e = np.array(['python', 'scala', 'java', 'c++'], dtype=np.string_)
e
```




    array([b'python', b'scala', b'java', b'c++'], dtype='|S6')




```python
e = np.array(['python', 'scala', 'java', 'c++'], dtype='S8')
e
```




    array([b'python', b'scala', b'java', b'c++'], dtype='|S8')



### 1.1.5 改变ndarray的形状
直接修改ndarray的shape值

使用reshape函数，可以创建一个改变了尺寸的新数组，原数组的shape保持不变，但注意他们共享内存空间，因此修改任何一个也对另一个产生影响，因此注意新数组的元素个数必须与原数组一样

当指定新数组某个轴的元素为-1时，将根据数组元素的个数自动计算此轴的长度


```python
a = np.arange(0, 20, 2)
print(a)
print(a.size)
```

    [ 0  2  4  6  8 10 12 14 16 18]
    10
    


```python
a.reshape(2, 5)
```




    array([[ 0,  2,  4,  6,  8],
           [10, 12, 14, 16, 18]])




```python
a.reshape(-1, 5)
```




    array([[ 0,  2,  4,  6,  8],
           [10, 12, 14, 16, 18]])




```python
b = a.reshape(2, -1)
b
```




    array([[ 0,  2,  4,  6,  8],
           [10, 12, 14, 16, 18]])




```python
b.shape
```




    (2, 5)




```python
b.shape = 5, -1
b
```




    array([[ 0,  2],
           [ 4,  6],
           [ 8, 10],
           [12, 14],
           [16, 18]])




```python
a
```




    array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18])




```python
a[1] = 3
b
```




    array([[ 0,  3],
           [ 4,  6],
           [ 8, 10],
           [12, 14],
           [16, 18]])



## 1.2 NumPy基本操作
数组与标量、数组之间的运算

数组的矩阵积（matrix product）

数组的索引与切片

数组转置与轴对换

通用函数：快速的元素级数组函数

聚合函数

np.where函数

np.unique函数

### 1.2.1 数组与标量、数组之间的运算
数组不用循环即可对每个元素执行批量运算，这通常就叫做矢量化，即用数组表达式代替循环的做法

矢量化数组运算性能要比纯Python方式快上一两个数量级

大小相等的数组之间的任何算术运算都会将运算应用到元素级


```python
import numpy as np
arr1 = np.array([1, 2, 3, 4, 5])
arr1 + 2
```




    array([3, 4, 5, 6, 7])




```python
arr1 - 2
```




    array([-1,  0,  1,  2,  3])




```python
arr1 * 2
```




    array([ 2,  4,  6,  8, 10])




```python
1/arr1
```




    array([1.        , 0.5       , 0.33333333, 0.25      , 0.2       ])




```python
1//arr1
```




    array([1, 0, 0, 0, 0], dtype=int32)




```python
arr1 ** 2
```




    array([ 1,  4,  9, 16, 25], dtype=int32)




```python
arr1 = np.array([[1, 2.0], [1.9, 3.4]])
arr2 = np.array([[3.6, 1.2], [2.0, 1.2]])
arr1 + arr2
```




    array([[4.6, 3.2],
           [3.9, 4.6]])




```python
arr1 - arr2
```




    array([[-2.6,  0.8],
           [-0.1,  2.2]])




```python
arr1 * arr2
```




    array([[3.6 , 2.4 ],
           [3.8 , 4.08]])




```python
arr1 / arr2
```




    array([[0.27777778, 1.66666667],
           [0.95      , 2.83333333]])



元素级运算

像上面例子展现出来的，加、减、乘、除、幂运算等，可以用于数组与标量、大小相等数组之间。

在Numpy中，大小相等的数组之间运算，为元素级运算，即只用于位置相同的元素之间，所得到的运算结果组成一个新的数组，运算结果的位置跟操作数位置相同
### 1.2.2 数组的矩阵积（matrix product）
两个二维矩阵（多维数组即矩阵）满足第一个矩阵的列数与第二个矩阵的行数相同，那么可以进行矩阵乘法，即矩阵积，矩阵积不是元素级的运算

两个矩阵相乘结果所得到的的数组中每个元素为，第一个矩阵中与该元素行号相同的元素与第二个矩阵中与该元素列号相同的元素，两两相乘后求和


```python
import numpy as np
arr = np.array([[120, 60, 220], [115, 23, 201], [132, 48, 230]])
arr2 = np.array([[12.34, 0.04], [204.56, 2.34], [9.89, 0.45]])
arr.dot(arr2)
```




    array([[15930.2 ,   244.2 ],
           [ 8111.87,   148.87],
           [13722.46,   221.1 ]])




```python
np.dot(arr, arr2)
```




    array([[15930.2 ,   244.2 ],
           [ 8111.87,   148.87],
           [13722.46,   221.1 ]])



### 1.2.3 数组的索引与切片
多维数组的索引

NumPy中数组的切片

布尔型索引

花式索引

多维数组的索引


```python
arr = np.array([[[2, 3, 4, 5], [1, 3, 4, 9]], [[0, 3, 4, 8], [2, 4, 9, 4]], [[1, 4, 5, 8], [2, 5, 6, 8]], [[2, 3, 6, 8], [3, 4, 8, 9]]])
arr.shape
```




    (4, 2, 4)




```python
arr[3]
```




    array([[2, 3, 6, 8],
           [3, 4, 8, 9]])




```python
arr[3][1]
```




    array([3, 4, 8, 9])




```python
arr[3][1][2]
```




    8



NumPy中数组的切片


```python
arr[1]
```




    array([[0, 3, 4, 8],
           [2, 4, 9, 4]])




```python
arr[1][0]
```




    array([0, 3, 4, 8])




```python
arr[1][0][1:3]
```




    array([3, 4])




```python
arr[1, :, 1:3]
```




    array([[3, 4],
           [4, 9]])



注意NumPy中通过切片得到的新数组，只是原来数组的一个视图，因此对新数组
进行操作也会影响原数组

布尔型索引


```python
A = np.random.random((4, 4))
A
```




    array([[0.88702666, 0.96635343, 0.59112595, 0.5539748 ],
           [0.27142736, 0.95130394, 0.37464197, 0.46125517],
           [0.357631  , 0.75003158, 0.67690867, 0.80261789],
           [0.08145381, 0.91772128, 0.25983881, 0.26314874]])




```python
A < 0.5
```




    array([[False, False, False, False],
           [ True, False,  True,  True],
           [ True, False, False, False],
           [ True, False,  True,  True]])




```python
A[A < 0.5]
```




    array([0.27142736, 0.37464197, 0.46125517, 0.357631  , 0.08145381,
           0.25983881, 0.26314874])



True的位置上的元素取出组成一个新的数组


```python
names = np.array(['Tom', 'Merry'])
names == 'Tom'
```




    array([ True, False])




```python
scores = np.array([[98, 87, 45, 34], [88, 45, 23, 98]])
scores[names == 'Tom']
```




    array([[98, 87, 45, 34]])




```python
scores[names == 'Tom', 2]
```




    array([45])




```python
scores[(names == 'Tom') | (names == 'Merry') ]
```




    array([[98, 87, 45, 34],
           [88, 45, 23, 98]])




```python
scores[~((names == 'Tom') & (names == 'Merry')) ]
```




    array([[98, 87, 45, 34],
           [88, 45, 23, 98]])



花式索引

花式索引（Fancy indexing）指的是利用整数数组进行索引


```python
arr = np.arange(32).reshape(8, 4)
arr
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19],
           [20, 21, 22, 23],
           [24, 25, 26, 27],
           [28, 29, 30, 31]])




```python
arr[[0, 3, 5]]
```




    array([[ 0,  1,  2,  3],
           [12, 13, 14, 15],
           [20, 21, 22, 23]])




```python
arr[[0, 3, 5], [0, 3, 2]]
```




    array([ 0, 15, 22])




```python
arr[np.ix_([0, 3, 5], [0, 2, 3])]
```




    array([[ 0,  2,  3],
           [12, 14, 15],
           [20, 22, 23]])



ix_函数产生一个索引器

### 1.2.4 数组转置与轴对换
transpose函数用于数组转置，对于二维数组来说就是行列互换

数组的T属性，也是转置


```python
arr.transpose()
```




    array([[ 0,  4,  8, 12, 16, 20, 24, 28],
           [ 1,  5,  9, 13, 17, 21, 25, 29],
           [ 2,  6, 10, 14, 18, 22, 26, 30],
           [ 3,  7, 11, 15, 19, 23, 27, 31]])




```python
arr.T
```




    array([[ 0,  4,  8, 12, 16, 20, 24, 28],
           [ 1,  5,  9, 13, 17, 21, 25, 29],
           [ 2,  6, 10, 14, 18, 22, 26, 30],
           [ 3,  7, 11, 15, 19, 23, 27, 31]])



### 1.2.5 通用函数：快速的元素级数组函数
ufunc：一种对ndarray中的数据执行元素级运算的函数，也可以看做是简单函数
（接受一个或多个标量值，并产生一个或多个标量值）的矢量化包装器


```python
arr = np.arange(10).reshape(2, -1)
arr
```




    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])




```python
np.sqrt(arr)
```




    array([[0.        , 1.        , 1.41421356, 1.73205081, 2.        ],
           [2.23606798, 2.44948974, 2.64575131, 2.82842712, 3.        ]])



### 常见的一元通用函数
|一元ufunc|说明|
|:-|:-|
|abs,fabs|计算整数、浮点数或复数的绝对值。对于非复数值，可以使用更快的fabs|
|sqrt|计算各元素的平方根，相当于 arr ** 0.5|
|square|计算各元素的平方，相当于arr ** 2|
|exp|计算各元素的指数e的x次方|
|log, log10, log2,log1p|分别为自然对数、底数是10的log，底数为2的log，log(1+x)|
|sign|计算各元素的正负号：1 正数，0 零 ， -1 负数|
|cell|计算各元素的ceiling值，即大于等于该值的最小整数|
|floor|计算各元素的floor值，即小于等于该值的最大整数|
|rint|将各元素值四舍五入到最接近的整数，保留dtype|
|modf|将数组的小数位和整数部分以两个独立数组的形式返回|
|isnan|返回一个表示“哪些值是NaN（不是一个数字）”的布尔类型数组|
|isfinite，isinf|分别返回一个表示“哪些元素是有穷的（非inf，非NaN）”或
“哪些元素是无穷的”的布尔型数组|
|cos、cosh、sin、sinh、tan、tanh|普通型和双曲型三角函数|
|arccos,arccosh,arcsin、arctan、arctanh|反三角函数|
|logical_not|计算各元素not x的真值，相当于 ~ 和 -arr|

### 1.2.6 聚合函数
* 聚合函数是对一组值（比如一个数组）进行操作，返回一个单一值作为结果的函
数。因此求数组所有元素之和、求所有元素的最大最小值以及标准差的函数就是
聚合函数


```python
arr = np.array([1.0, 2.0, 3.0, 4.0])
arr.max()
```




    4.0




```python
arr.min()
```




    1.0




```python
arr.mean()
```




    2.5




```python
arr.std()
```




    1.118033988749895




```python
np.sqrt(np.power(arr - arr.mean(), 2).sum()/arr.size)
```




    1.118033988749895



聚合函数可以指定对数值的某个轴元素进行操作


```python
arr = np.array([[1, 2, 3, 4], [3, 4, 5, 6]])
arr
```




    array([[1, 2, 3, 4],
           [3, 4, 5, 6]])




```python
arr.mean(axis=0)  # 对同一列上的元素进行聚合
```




    array([2., 3., 4., 5.])




```python
arr.mean(axis=1)  # 对同一行上的元素进行聚合
```




    array([2.5, 4.5])




```python
arr.sum(axis=0)
```




    array([ 4,  6,  8, 10])




```python
arr.max(axis=1)
```




    array([4, 6])



### 1.2.7 np.where函数
* np.where函数是三元表达式 x if condition else y的矢量化版本


```python
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
condition = np.array([True, False, True, True, False])
result = [(x if c else y) for x,y,c in zip(xarr, yarr, condition)]
result
```




    [1.1, 2.2, 1.3, 1.4, 2.5]




```python
result = np.where(condition, xarr, yarr)
result
```




    array([1.1, 2.2, 1.3, 1.4, 2.5])



* 将数组中所有NaN缺失值替换为0


```python
arr = np.array([[1, 2, np.NAN, 4], [3, 4, 5, np.NAN]])
arr
```




    array([[ 1.,  2., nan,  4.],
           [ 3.,  4.,  5., nan]])




```python
np.isnan(arr)
```




    array([[False, False,  True, False],
           [False, False, False,  True]])




```python
np.where(np.isnan(arr), 0, arr)
```




    array([[1., 2., 0., 4.],
           [3., 4., 5., 0.]])



### 1.2.8 np.unique函数

求数组中不重复的元素


```python
arr = np.array([1.0, 2.0, 1.0, 4.0])
np.unique(arr)
```




    array([1., 2., 4.])



### 1.2.9 数组数据文件读写

将数组以二进制格式保存到磁盘

存取文本文件


```python
data = np.array([[1, 2, 3, 4], [3, 4, 5, 6]])
np.save('data', data)
np.load('data.npy')
```




    array([[1, 2, 3, 4],
           [3, 4, 5, 6]])




```python

```
