---
title: 中文分词--HMM算法
date: 2021-8-16
tags: 
  - NLP
  - 分词
  - HMM
author: zhuzongfa
location: Shanghai 
---

上一篇文章中，我们讲了如果用查词典的方式对中文语句分词，但这种方式不能百分百地解决中文分词问题，比如对于未登录词（在已有的词典中，或者训练语料里面没有出现过的词），无法用查词典的方式来切分，这时候可以用隐马尔可夫模型（HMM）来实现。在实际应用中，一般也是将词典匹配分词作为初分手段，再利用其他方法提高准确率。

<br/>

### HMM介绍

**隐马尔可夫模型**（Hidden Markov Model，HMM）是统计模型，是关于**时序**的概率图模型，它用来描述一个含有隐含未知参数的马尔可夫过程，即由一个隐藏的马尔可夫链随机生成不可观测的状态随机序列，再由各个状态生成一个观测而产生观测随机序列的过程。序列的每一个位置又可以看作是一个**时刻**，其结构见下图。其难点是从可观察的参数中确定该过程的隐含参数，然后利用这些参数来作进一步的分析，例如中文分词。

![HMM](https://i.loli.net/2021/08/16/6rhTxPACwvjOY9d.png)

如上图所示，状态序列H可表示为：
$$
H = H_{1},H_{2},...,H_{T}
$$
假设总共有n个状态，即每个状态序列必为状态集合之一，状态值集合Q为：
$$
Q=\{ q_{1},q_{2},...,q_{n} \}
$$


观测序列O表示为：
$$
O=O_{1},O_{2},...,O_{T}
$$
假设观测值总共有m个，则观测值集合为：
$$
V=\{ v_{1},v_{2},...,v_{m} \}
$$

<br/>

#### 一个模型，两个假设，三个问题

**1、一个模型**

HMM的基本元素可以表示为
$$
\lambda = \{ Q, V, \pi, A, B \}
$$
Q：状态值集合

V： 观测值集合

π：初始概率分布

A：$[a_{ij}]$ 状态转移矩阵

B：$[b_{j(k)}]$ 给定状态下，观测值概率矩阵，即发射矩阵

**2、两个假设**

- 齐次Markov

即假设观测序列中t时刻的状态，只跟上一时刻t-1有关，
$$
P(h_{t+1}|h_{t},...,h_{1}；o_{t},...,o_{1}) = P(h_{t+1}|h_{t})
$$


- 观测独立

即每个时刻的观测值只由该时刻的状态值决定
$$
P(o_{t}|o_{t-1},...,o_{1};h_{t},...,h_{1})=P(o_{t}|h_{t})
$$
**3、三个问题**

HMM在实际应用中主要用来解决3类问题:
- 评估问题(概率计算问题)
  即给定观测序列 $$O=O_{1},O_{2},O_{3}…O_{t}$$和模型参数λ=(A,B,π)，怎样有效计算这一观测序列出现的概率.
  (Forward-backward算法)

- 解码问题(预测问题)
  即给定观测序列$$O=O_{1},O_{2},O_{3}…O_{t}$$和模型参数λ=(A,B,π)，怎样寻找满足这种观察序列意义上最优的隐含状态序列S。

   (viterbi算法，近似算法)

- 学习问题
  即HMM的模型参数λ=(A,B,π)未知，如何求出这3个参数以使观测序列$$O=O_{1},O_{2},O_{3}…O_{t}$$的概率尽可能的大.
   (即用极大似然估计的方法估计参数，Baum-Welch,EM算法)

<br/>

### HMM模型在中文分词中的应用

在中文分词中，我们经常用sbme的字标注法来标注语料，即通过给每个字打上标签来达到分词的目的，其中s表示single，代表单字成词；b表示begin，代表多字词的开头；m表示middle，代表三个及以上字词的中间部分；e表示end，表示多字词的结尾。这样`我是中国人`就可以表示为`ssbme`。

对于字标注的分词方法来说，输入就是n个字，输出就是n个标签。我们用$$λ=λ_{1}λ_{2}…λ_{n}$$表示输入的句子，$$o=o_{1}o_{2}…o_{n}$$表示输出。那最优的输出从概率的角度来看，就是求条件概率
$$
P(o|\lambda)=P(o_{1},...,O_{n}|\lambda_{1},...,\lambda_{n})
$$
即要求上式概率最大，根据独立性假设，即每个字的输出只与当前字有关，上式可表示为：
$$
P(o_{1},...,O_{n}|\lambda_{1},...,\lambda_{n}) = \prod \limits_{k}P(o_{k}|\lambda_{k}), k=1,...,n
$$
由bayes定理，
$$
P(o|\lambda) = \frac{P(\lambda, o)}{P(\lambda)}=\frac{P(o)P(\lambda/o)}{P(\lambda)}\sim P(o)P(\lambda/o)
$$
同样地由独立性假设，
$$
P(\lambda|o) = \prod \limits_{k}P(\lambda_{k}|o_{k}), k=1,...,n
$$
由HMM的markov假设：每个输出仅和上一个时刻相关，有
$$
P(o) = P(o_{1})P(o_{2}|o_{1})P(o_{3}|o_{2})...P(o_{n}|o_{n-1})
$$
因此，最初的求解可以转化为
$$
P(o|\lambda)\sim P(o)P(\lambda/o) = P(o_{1})P(\lambda_{1}|o_{1}) P(o_{2}|o_{1})P(\lambda_{2}|o_{2})P(o_{3}|o_{2})...P(o_{n}|o_{n-1})P(\lambda_{n}|o_{n})
$$
其中$$P(\lambda_{k}|o_{k})$$为发射概率，$$P(o_{k}|o_{k-1})$$为转移概率

<br/>

### 代码实现

用HMM模型来实现中文分词，状态值集合有四种，分别是 s, b, m, e；观测值集合，即所有汉子；转移状态集合

有`['ss', 'sb', 'bm', 'be', 'mm', 'me', 'es', 'eb']`, 其他组合都为0，即不可能的情况，比如`bs, ms`这些状态不可能转移。这里从jieba分词的github下载了一份字典库`dict.txt.small`，需要的朋友可以去官方github下载。我们先看看该词典的格式

```shell
[zhuzf@localhost test]$ head -n 5 dict.txt.small 
的 3188252 uj
了 883634 ul
是 796991 v
在 727915 p
和 555815 c
[zhuzf@localhost test]$ tail -n 5 dict.txt.small
龙胜 10 nr
龙蛇混杂 10 i
龙骨山 10 nr
龚松林 10 nr
龟板 10 n
```

我们根据这份词典统计每个字是s,b,m,e的发射概率， 计算方法为将词典中所有的一字词都计入s标签下，把多字词的首字都计入b标签下，最后一个字计入e标签下，其余的计入m标签中。此外计算过程中还使用了对数概率，防止溢出；

```python
hmm_model = {i: Counter(i) for i in 'sbme'}
with open('data/dict.txt.small', encoding='utf-8') as f:
    for line in f:
        lines = line.strip('\n').split(' ')
        if len(lines[0]) == 1:
            hmm_model['s'][lines[0]] += int(lines[1])
        else:
            hmm_model['b'][lines[0][0]] += int(lines[1])
            hmm_model['e'][lines[0][-1]] += int(lines[1])
        for m in lines[0][1:-1]:
            hmm_model['m'][m] += int(lines[1])
```

转移概率，根据结巴分词的状态转移矩阵`prob_trans.p`得到， value值为概率对数

```python
P={'B': {'E': -0.510825623765990, 'M': -0.916290731874155},
 'E': {'B': -0.5897149736854513, 'S': -0.8085250474669937},
 'M': {'E': -0.33344856811948514, 'M': -1.2603623820268226},
 'S': {'B': -0.7211965654669841, 'S': -0.6658631448798212}}
trans = {}
for key, values in P.items():
    for k,v in values.items():
        trans[(key+k).lower()] = v 
print(trans)
```

output

```python
{'be': -0.51082562376599,
 'bm': -0.916290731874155,
 'eb': -0.5897149736854513,
 'es': -0.8085250474669937,
 'me': -0.33344856811948514,
 'mm': -1.2603623820268226,
 'sb': -0.7211965654669841,
 'ss': -0.6658631448798212}
```

viterbi算法， 动态规划算法实现，得到最优路径，即得到概率最大的sbme组合， 对于概率估算，简单采用了加1平滑法，没出现的单字都算1次。

```python
def viterbi(nodes, trans):
    paths = nodes[0]  # 初始状态
    for l in range(1, len(nodes)): 
        paths_ = paths 
        paths = {}
        for i in nodes[l]:
            nows = {}
            for j in paths_:
                if j[-1]+i in trans:
                    nows[j+i] = paths_[j]+nodes[l][i]+trans[j[-1]+i]

            nows_values_list = list(nows.values())
            k = nows_values_list.index(max(nows_values_list))

            nows_keys_list = list(nows.keys())
            paths[nows_keys_list[k]] = nows_values_list[k]

    paths_values_list = list(paths.values())
    paths_keys_list = list(paths.keys())

    return paths_keys_list[paths_values_list.index(max(paths_values_list))]
```

HMM模型，对输入句子做分词

```python
def hmm_cut(s):
    nodes = [{i: log(j[t]+1)-log_total[i]
                for i, j in hmm_model.items()} for t in s]
    tags = viterbi(nodes, trans)
    print(tags)
    words = [s[0]]
    for i in range(1, len(s)):
        if tags[i] in ['b', 's']:
            words.append(s[i])
        else:
            words[-1] += s[i]
    return words
```

测试

```
text = '华为手机深得大家的喜欢'
print' '.join(hmm_cut(text))）
# '华为 手机 深得 大家 的 喜欢'
text = '王五的老师经常夸奖他'
print(' '.join(hmm_cut(text)))
# '王五 的 老师 经常 夸奖 他'
```

可见对于未登录词，王五HMM模型可以发现这是个人名，分词时组合在一起，而查字典方法做不到。

<br/>

*文章参考：*

*1、https://blog.csdn.net/sinat_33741547/article/details/78690440*

*2、https://github.com/ustcdane/annotated_jieba/blob/master/jieba/finalseg/__init__.py*

*3、https://spaces.ac.cn/archives/3922*











