---
title: jupyter notebook的美化
date: 2021-8-11
tags: 
  - jupyter notebook
author: zhuzongfa
location: Shanghai 
---


### 一、背景

jupyter notebook默认是没有主题可选择的，有的系统安装后默认配置使用起来不太舒服，比如默认字体对代码阅读不友好，或者喜欢给代码加行号的也不知道该去哪里设置。本篇想介绍一下jupyter notebook的优化。



### 二、stylus管理器

stylus是一个网页插件，可对所有网页的样式作修改。同时也可以配置样式给指定的网页，比如我这里

```python
#notebook-container * {
font-family: Consolas, "微软雅黑"
}
```

notebook的字体就改为微软雅黑了。当然还可以加其他的样式优化代码，大家可以随意发挥。



### 三、主题

主题皮肤有一个第三方库jupyterthemes网上推荐比较多，但是本人使用后觉得有些地方改得有些生硬。所以最终没有采纳这种方案。

我使用的是在原来的基础上修改CSS样式，即在`.jypyter/`下放入`custom`文件夹，里面是`custom.css`文件。我比较喜欢黑色的主题，所以在网上找了一个custom.css样式的文件下载。

但是我个人感觉直接把这个放在文件夹下不太方便，换了一个电脑又要重新下载一遍。所以我把`custom.css`里的内容复制到了stylus管理器中的样式文档中。这样无论我什么时候使用，只要用到我的chrome浏览器，就可以美化notebook了。



### 四、*nbextensions* 插件

notebook插件可以用于增强用户体验并提供多种个性化技术。使用 *nbextensions* 库可以用来安装所有必需的小部件。该库利用不同的Javascript模型来丰富笔记本前端。

```python
! pip install jupyter_contrib_nbextensions
! jupyter contrib nbextension install --system
```

一旦 *nbextensions* 安装好，你会发现，在Jupyter notebook主页（下图），会有一个额外的标签。

![插件](https://i.loli.net/2021/08/11/WgZTm3iwVbQcOfB.png "插件")

通过单击Nbextensions选项卡，将为我们提供可用小部件的列表。比如加入行号、代码自动补全等等这里都有，可以多去尝试里面的各种小功能。

![小组件](https://i.loli.net/2021/08/11/NXEivabekgT19Bw.png "插件")

最后的效果如下：

![效果](https://i.loli.net/2021/08/11/rOkjIyezwS3ZHbl.png "dark效果")

