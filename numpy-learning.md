
#1 NumPy
1.NumPy系统是Python的一种开源的数值计算扩展，是一个用python实现的科学计算包.

2.NumPy（Numeric Python）提供了许多高级的数值编程工具，如：矩阵数据类型、矢量处理，以及精密的运算库。专为进行严格的数字处理而产生。

3.使用Anaconda发行版的Python，已经帮我们事先安装好了Numpy模块，因此无需另外安装

4.依照标准的Numpy约定，习惯使用 import numpy as np方式导入该模块

##1.1 Numpy的数据结构：ndarry，一种多维数组对象
###1.1.1 ndarray介绍
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



###1.1.2 ndarray的常见创建方式
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



###1.1.3 ndarray的其他创建方式
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




    array([[[0.43582991, 0.73243019, 0.14293359, 0.79835257],
            [0.34472292, 0.28618313, 0.25359567, 0.4544699 ]],
    
           [[0.0913177 , 0.09678176, 0.61065844, 0.8430518 ],
            [0.00939408, 0.52705446, 0.34309223, 0.10155234]]])



###1.1.4 NumPy中的数据类型
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



###1.1.5 改变ndarray的形状
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



##1.2 NumPy基本操作
数组与标量、数组之间的运算

数组的矩阵积（matrix product）

数组的索引与切片

数组转置与轴对换

通用函数：快速的元素级数组函数

聚合函数

np.where函数

np.unique函数

###1.2.1 数组与标量、数组之间的运算
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
###1.2.2 数组的矩阵积（matrix product）
两个二维矩阵（多维数组即矩阵）满足第一个矩阵的列数与第二个矩阵的行数相同，那么可以进行矩阵乘法，即矩阵积，矩阵积不是元素级的运算

两个矩阵相乘结果所得到的的数组中每个元素为，第一个矩阵中与该元素行号相同的元素与第二个矩阵中与该元素列号相同的元素，两两相乘后求和


```python

```
