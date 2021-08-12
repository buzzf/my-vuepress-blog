---
title: 向量基础
date: 2021-8-11
tags: 
  - 线性代数
  - 计算机图形学
author: yangzhenyu
location: beijing  
---

## 向量的定义

一个同时具有大小和方向的量就是向量。可以形象的表示为带箭头的线段。

可以记为 $\vec{a}$​​​​ 或者 ***a***

代数表示：

在指定坐标系S中，基向量为：$\vec{e_1},\vec{e_2},...,\vec{e_n}$ 。

 $\vec{a}= a_1\vec{e_1} + a_2\vec{e_2} + ... + a_n\vec{e_n}$

其中$a_1,a_2,...,a_n$分别为在$\vec{e_1},\vec{e_2},...,\vec{e_n}$​​​下的投影。当基向量已知，可以省略基向量，表示为：

$\vec{a} = (a_1,a_2,...,a_n)$

在常见的三维空间直角坐标系$O_{xyz}$​里，x轴、y轴、z轴的单位向量分别为$\vec{i}、\vec{j}、\vec{k}$任意向量可表示为：

$\vec{a} = (x,y,z) = x\vec{i} + y\vec{j} + z\vec{k}$

在矩阵运算中，向量通常被写成类似矩阵的列向量或行向量。在线性代数中所指的向量默认为列向量。如一个向量 $\vec{a} = (x,y,z)$​​​ 可写成：

$***a*** = \begin{bmatrix} x \\ y \\ z \end{bmatrix}$​​

$\vec{a}^T = \begin{bmatrix}x&y&z\end{bmatrix}$​​​​​​​​​​



## 向量的运算

向量的加法：

$\vec{a} = x_a\vec{i} + y_a\vec{j} + z_a\vec{k} = (x_a,y_a,z_a)$​​​​ 

$\vec{b} = x_b\vec{i} + y_b\vec{j} + z_b\vec{k} = (x_b,y_b,z_b)$​​ 

则：

$\vec{a} + \vec{b} = (x_a + x_b)\vec{i} + (y_a + y_b)\vec{j} + (z_a + z_b)\vec{k} = (x_a + x_b,y_a+y_b,z_a+z_b)$​​



数量积：



叉乘：
