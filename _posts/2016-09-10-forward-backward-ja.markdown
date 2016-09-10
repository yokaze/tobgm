---
permalink: /ja/forward-backward.html
layout: post
title: "Forward-Backward アルゴリズム"
---
Forward-Backward アルゴリズムは， [隠れマルコフ分布][hmmdist] にしたがう $$Z$$ の同時分布
$$p(Z) = \mathcal{H} \left( Z \middle| \pi, A, E \right)$$ から，
$$z_n$$ の周辺分布

$$p(z_n)　= \sum_{z_1 \cdots z_{n-1}, z_{n+1} \cdots z_N} p(Z)$$

と，となりあう二変数の同時分布

$$p(z_n, z_{n+1}) = \sum_{z_1 \cdots z_{n-1}, z_{n+2} \cdots z_N} p(Z)$$

を計算するアルゴリズムである．

## Forward アルゴリズム

隠れマルコフ分布にしたがう $$Z$$ の同時分布は以下のように書ける．

$$
\mathcal{H} \left(Z \middle| \pi, A, E \right) = \frac{1}{\mathcal{Z}} \prod_{k=1}^K \pi_k^{z_{1k}} \prod_{m=1}^{N-1} \prod_{jk} A_{jk}^{z_{mj}z_{m+1,k}} \prod_{mk} e_{mk}^{z_{mk}}
$$

$$\mathcal{Z}$$ は正規化定数である．

非正規化された隠れマルコフ分布 $$\mathcal{H}^-$$ と，$$\alpha$$-Message を以下の通り定義する．

$$
\begin{align}
\mathcal{H}^- \left(Z \middle| \pi, A, E \right) &= \prod_{k=1}^K \pi_k^{z_{1k}} \prod_{m=1}^{N-1} \prod_{jk} A_{jk}^{z_{mj}z_{m+1,k}} \prod_{mk} e_{mk}^{z_{mk}} \\
\mathcal{H}^- \left( z_1, \cdots, z_n \middle| \pi, A, E \right) &= \prod_{k=1}^K \pi_k^{z_{1k}} \prod_{m=1}^{n-1} \prod_{jk} A_{jk}^{z_{mj}z_{m+1,k}} \prod_{m=1}^n \prod_{k=1}^K e_{mk}^{z_{mk}} \\
\alpha_{nk} &= \left[ \sum_{z_1 \cdots z_{n-1}} \mathcal{H}^- \left( z_1, \cdots, z_n \middle| \pi, A, E\right) \right]_{z_{nk} = 1}
\end{align}
$$

$$\mathcal{H}^-$$ について

$$
\mathcal{H}^- \left( z_1, \cdots, z_m \middle| \pi, A, E \right) = \mathcal{H}^- \left( z_1, \cdots, z_{m-1} \middle| \pi, A, E \right) \prod_{jk} A_{jk}^{z_{m-1,j} z_{mk}} \prod_{k=1}^{K} e_{mk}^{z_{mk}}
$$

が成り立つため，$$\alpha_{nk}$$ と $$\mathcal{Z}$$ は以下のように計算できる．

$$
\begin{align}
\alpha_{1k} &= \left[ \mathcal{H}^- \left( z_1 \middle| \pi, A, E\right) \right]_{z_{1k} = 1} \\
&= \left[ \prod_{k'=1}^K \pi_{k'}^{z_{1k'}} e_{1k'}^{z_{1k'}} \right]_{z_{1k} = 1} \\
&= \pi_k e_{1k} \\
\alpha_{nk} &= \left[ \sum_{z_1 \cdots z_{n-1}} \mathcal{H}^- \left( z_1, \cdots, z_n \middle| \pi, A, E\right) \right]_{z_{nk} = 1} \\
&= \left[ \sum_{z_{n-1}} \sum_{z_1 \cdots z_{n-2}} \mathcal{H}^- \left( z_1, \cdots, z_{n-1} \middle| \pi, A, E \right) \prod_{j'k'} A_{j'k'}^{z_{n-1,j'} z_{nk'}} \prod_{k'=1}^{K} e_{nk'}^{z_{nk'}} \right]_{z_{nk} = 1} \\
&= \left[ \sum_{j=1}^K \left[ \sum_{z_1 \cdots z_{n-2}} \mathcal{H}^- \left( z_1, \cdots, z_{n-1} \middle| \pi, A, E \right) \right]_{z_{n-1,j}=1} \prod_{j'k'} A_{j'k'}^{z_{n-1,j'} z_{nk'}} \prod_{k'=1}^{K} e_{nk'}^{z_{nk'}} \right]_{z_{nk} = 1} \\
&= \sum_{j=1}^K \alpha_{n-1,j} \left[ \prod_{j'k'} A_{j'k'}^{z_{n-1,j'} z_{nk'}} \prod_{k'=1}^{K} e_{nk'}^{z_{nk'}} \right]_{z_{n-1,j}=1,z_{nk}=1} \\
&= e_{nk} \sum_{j=1}^K \alpha_{n-1,j} A_{jk} \\
\mathcal{Z} &= \sum_{k=1}^K \alpha_{Nk}
\end{align}
$$

## Backward アルゴリズム

$$\beta$$-Message を以下の通り定義する．

$$
\begin{align}
\beta_{nj} &= \left[ \frac{1}{\alpha_n} \sum_{\neg z_n} \mathcal{H}^- \left(Z \middle| \pi, A, E \right) \right]_{z_{nj} = 1} \\
&= \left[ \frac{\sum_{z_1 \cdots z_{n-1}} \sum_{z_{n+1} \cdots z_N} \prod_{k'=1}^K \pi_{k'}^{z_{1k'}} \prod_{m=1}^{N-1} \prod_{j'k'} A_{j'k'}^{z_{mj'}z_{m+1,k'}} \prod_{mk'} e_{mk'}^{z_{mk'}}}
{\sum_{z_1 \cdots z_{n-1}}\prod_{k'=1}^K \pi_{k'}^{z_{1k'}} \prod_{m=1}^{n-1} \prod_{j'k'} A_{j'k'}^{z_{mj'}z_{m+1,k'}} \prod_{m=1}^n \prod_{k'=1}^K e_{mk'}^{z_{mk'}}} \right]_{z_{nj} = 1} \\
&= \left[ \sum_{z_{n+1} \cdots z_N} \prod_{m=n}^{N-1} \prod_{j'k'} A_{j'k'}^{z_{mj'} z_{m+1,k'}} \prod_{m=n+1}^N \prod_{k'=1}^K e_{mk'}^{z_{mk'}} \right]_{z_{nj} = 1}
\end{align}
$$

$$
\beta_{mj}
$$
は以下のように計算できる．

$$
\begin{align}
\beta_{Nj} &= 1 \\
\beta_{mj} &= \sum_{k=1}^K e_{m+1,k} \beta_{m+1,k} A_{jk}
\end{align}
$$

## 周辺分布の計算

$$
\begin{align}
p(z_n) &= \sum_{\neg z_n} \mathcal{H} \left(Z \middle| \pi, A, E \right) \\
&=\frac{\alpha_n \beta_n}{\mathcal{Z}} \\
\left[ p(z_n, z_{n+1}) \right]_{jk} &= \left[ \sum_{\neg z_n, z_{n+1}} \mathcal{H} \left(Z \middle| \pi, A, E \right) \right]_{z_n = j, z_{n+1} = k} \\
&= \frac{\alpha_{nj} e_{n+1,k} A_{jk} \beta_{n+1,k}}{\mathcal{Z}}
\end{align}
$$

## ソースコード
各変数の桁数が非常に大きくなるため，計算はすべて対数領域でおこなう．

{% highlight python %}
from scipy import exp, float64, log, rand, zeros
from scipy.misc import logsumexp

# lem: Log Emission Probability
def forward_backward(lpi, lA, lem):
    num, ncl = lem.shape

    # Forward Calculation
    la = zeros([num, ncl], dtype = float64)
    la[0] = lpi + lem[0]
    for i in xrange(1, num):
        for k in xrange(ncl):
            la[i, k] = lem[i, k] + logsumexp(la[i - 1, :] + lA[:, k])

    # Backward Calculation
    lb = zeros([num, ncl], dtype = float64)
    for irev in xrange(1, num):
        index = num - irev - 1
        for k in xrange(ncl):
            lb[index, k] = logsumexp(lem[index + 1, :] + lb[index + 1, :] + lA[k, :])

    return la, lb

def calc_lx_lr_lxi(lA, lem, la, lb):
    ncl = lA.shape[0]

    lX = logsumexp(la[-1])
    lr = la + lb - lX
    lxi = la[:-1, :, None] + lem[1:, None, :] + lA[None, :, :] + lb[1:, None, :] - lX
    return lX, lr, lxi

if (__name__ == '__main__'):
    from matplotlib import pyplot as pl
    num = 100
    ncl = 4
    pi = rand(ncl)
    pi /= pi.sum()
    lpi = log(pi)
    A = rand(ncl, ncl)
    A /= A.sum(1)[:, None]
    lA = log(A)
    lem = rand(num, ncl)

    la, lb = forward_backward(lpi, lA, lem)
    lX, lr, lxi = calc_lx_lr_lxi(lA, lem, la, lb)

    pl.imshow(exp(lr).T, aspect = 'auto', interpolation = 'nearest', cmap = 'Blues')
    pl.show()
{% endhighlight %}

## 参考文献

1. C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006.
2. M. J. Beal, *Variational Algorithms for Approximate Bayesian Inference*, PhD. Thesis, Gatsby Computational Neuroscience Unit, University College London, 2003.

[hmmdist]:/ja/pdf.html
