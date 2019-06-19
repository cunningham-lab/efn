---
title: Built-in Exponential Families
permalink: /families/
---

<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-MML-AM_CHTML' async></script>

<div class="topnav">
  <a class="active" href="../">Home</a>
  <a href="#">Exp Fams Library</a>
</div>

# [Built-in Exponential Families](https://github.com/cunningham-lab/tf_util/blob/master/tf_util/families.py) #

## Tractable exponential families ##

$$p(x) = \frac{h(x)}{A(\eta)} \exp\left\{ \eta(\theta)^\top t(x) \right\} = \frac{1}{A(\eta)} \exp\left\{ \begin{bmatrix} \eta(\theta) \\ 1 \end{bmatrix}^\top \begin{bmatrix} t(x) \\ \log(h(x)) \end{bmatrix} \right \}$$

| exponential family | $$\theta$$ | $$\eta(\theta)$$ | $$t(x)$$ | $$\log(h(x))$$ |
|---|---|---|---|---|
| Normal | $$\mu, \Sigma$$ | $$\begin{bmatrix} \Sigma^{-1}\mu \\ \frac{1}{2}\Sigma^{-1} \end{bmatrix}$$ | $$\begin{bmatrix} x \\ xx^\top \end{bmatrix}$$ | $$-\frac{D}{2}\log(2\pi)$$ |
| Dirichlet | $$\alpha$$ | $$\alpha$$ | $$\log(x)$$ | $$-\sum_{i=1}^D \log(x_i)$$ |
| Inv-Wishart | $$\Psi, m$$ | $$\begin{bmatrix} -\frac{1}{2}\Psi \\ \frac{-m+p+1}{2} \end{bmatrix}$$ | $$\begin{bmatrix} X^{-1} \\ \log(\|X\|) \end{bmatrix}$$ | 0 |

## Intractable exponential families ##

With an exponential family likelihood

$$ p(x_i|z) = \frac{1}{A(z)} \exp\left\{ \nu(z)^\top t(x_i) \right \} $$

and prior

$$ p_0(z) = \frac{1}{A_0(\alpha)} \exp\left\{ \alpha^\top t_0(z) \right\} $$

the posterior has the form:

$$ p(z | x_1,...,x_N) = p(z) \prod_{i=1}^N p(x_i \mid z) \propto  \exp\left\{ \begin{bmatrix} \alpha \\ \sum_i t(x_i) \\ -N \end{bmatrix}^\top\begin{bmatrix} t_0(z) \\ \nu(z) \\ \log A(z) \end{bmatrix} \right\} $$

| exponential family | $$\alpha$$ | $$t_0(z)$$ | $$\nu(z)$$ | $$t(x_i)$$ | $$\log(A(z))$$ |
|---|---|---|---|---|---|
| Hierarchical Dirichlet | $$\alpha - 1$$ | $$\log(z)$$ | $$\beta z$$ | $$\log(x_i)$$ | $$\log(B(\beta z))$$ |
| Dirichlet multinomial (N=1) | $$\alpha - 1$$ | $$\log(z)$$ | $$\log(z)$$ | $$x_i$$ | 0 |
| Truncated normal Poisson | $$\begin{bmatrix} \Sigma^{-1}\mu \\ \frac{1}{2}\Sigma^{-1} \end{bmatrix}$$ | $$\begin{bmatrix} x \\ xx^\top \end{bmatrix}$$ | $$\log(z)$$ | $$x_i$$ | z |
| Log-Gaussian Poisson | $$\begin{bmatrix} \Sigma^{-1}\mu \\ \frac{1}{2}\Sigma^{-1} \end{bmatrix}$$ | $$\begin{bmatrix} x \\ xx^\top \end{bmatrix}$$ | $$z$$ | $$x_i$$ | $$\exp(z)$$ |


