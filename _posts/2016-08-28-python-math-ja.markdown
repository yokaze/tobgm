---
layout: post
title: "Python による科学計算"
permalink: /ja/python-math.html
---

## スクリプト実行と対話シェル

Python がインストールされていれば，ファイル名を指定してスクリプトを実行できる．

{% highlight text %}
$ python helloworld.py
Hello World!
$
{% endhighlight %}

または，ファイル名を指定せずに実行することで対話シェルを起動できる．

{% highlight text %}
$ python
>>> print 'Hello World!'
Hello World!
>>>
{% endhighlight %}

## 基本的な構文

Python の変数は使用時に割り当てられる．
各変数は厳密な型を持つ．
{% highlight python %}
>>> x = 23
>>> type(x)
<type 'int'>
>>> x = 23.
>>> type(x)
<type 'float'>
{% endhighlight %}

`if`, `for` などの制御ブロックはインデントで表現する．
{% highlight python %}
if x == 20:
    print 'x is 20'
elif x == 30:
    print 'x is 30'
else:
    print 'x is unknown'

for i in xrange(20):
    print i
{% endhighlight %}

## 関数の定義

関数は `def` キーワードで定義できる．
{% highlight python %}
>>> def test(x):
...     return x + 20
...
>>> test(30)
50
{% endhighlight %}

## リスト，タプル，辞書
Python にはリスト，タプル，辞書が存在する．
{% highlight python %}
>>> x = [1, 2, 3, 4, 5]
>>> x[2]
3
>>> x = (1, 2, 3, 4, 5)
>>> x[2]
3
>>> x = { 'x': 20, 'y': 30 }
>>> x
{'y': 30, 'x': 20}
{% endhighlight %}

## リフレクション

`dir` 関数を使うことで変数のメンバをすべて表示することができる．
{% highlight python %}
>>> x = "test"
>>> dir(x)
['__add__', '__class__', '__contains__', '__delattr__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__getnewargs__', '__getslice__', '__gt__', '__hash__', '__init__', '__le__', '__len__', '__lt__', '__mod__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__rmod__', '__rmul__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '_formatter_field_name_split', '_formatter_parser', 'capitalize', 'center', 'count', 'decode', 'encode', 'endswith', 'expandtabs', 'find', 'format', 'index', 'isalnum', 'isalpha', 'isdigit', 'islower', 'isspace', 'istitle', 'isupper', 'join', 'ljust', 'lower', 'lstrip', 'partition', 'replace', 'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit', 'rstrip', 'split', 'splitlines', 'startswith', 'strip', 'swapcase', 'title', 'translate', 'upper', 'zfill']
{% endhighlight %}

## 科学計算
科学計算には `SciPy` と `matplotlib` が使用できる．
{% highlight python %}
>>> import scipy
>>> from matplotlib import pyplot as pl
>>> x = scipy.rand(5)
>>> x
array([ 0.15050235,  0.76100653,  0.21012437,  0.47341794,  0.49101636])
>>> pl.plot(x)
[<matplotlib.lines.Line2D object at 0x7f38f3eac990>]
>>> pl.show()
{% endhighlight %}

## スライス
Python にはスライス演算が存在し，リストの部分列が取得できる．部分列は\[始点:終点:ステップ幅\]であらわす．
{% highlight python %}
>>> from scipy import arange
>>> x = arange(10)
>>> x
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> x[2::3]
array([2, 5, 8])
{% endhighlight %}

## ブロードキャスト
`SciPy` ではテンソルの要素積の計算にブロードキャスト演算を使用する．
`NxMx1` のテンソルと `Nx1xK` のテンソルの積は `NxMxK` のテンソルになる．
`scipy.newaxis` を使うこともできるが， `newaxis` の実体は `None` である．
{% highlight python %}
>>> from scipy import rand
>>> x = rand(20, 30)
>>> y = rand(20, 40)
>>> x.shape
(20, 30)
>>> y.shape
(20, 40)
>>> x[:, :, None].shape
(20, 30, 1)
>>> y[:, None, :].shape
(20, 1, 40)
>>> (x[:, :, None] * y[:, None, :]).shape
(20, 30, 40)
{% endhighlight %}
