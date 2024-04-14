# bayesclean

#### 介紹
BClean清洗系统的源代码，清洗流程图如下所示。

![](https://gitee.com/wx_389ac836ab/bayesclean/raw/master/figure/overview.jpg)

#### 模块介绍

1. BClean文件，这是清洗系统的主文件，用于接收用户传递的参数，定义了核心的函数，包含结构生成、参数估计以及推理等调用函数，目的是调用各模块的核心类
2. analysis文件，用于评估清洗结果的精确率、召回率以及运行时间
3. src文件夹，包括用户约束（UC）类、贝叶斯网络结构类、补偿得分类、推理策略类等
4. example文件夹，包含对每个数据集编写的BClean流程代码
5. dataset数据集，存放测试数据集，其中真实数据集中有添加噪声的py文件
6. baseline文件夹，存放对比方法源代码及其论文

#### 用户约束（UC）编写方式
1. 初始化UC：

uc = UC(dirty_data)

2. 查看每一列推荐的正则表达式：

uc.PatternDiscovery()

3. 为特定属性定义UC的方式：

uc.build(attr = "birthyear", type = "Categorical", min_v = 4, max_v = 4, null_allow = "N", repairable = "Y", pattern = re.compile(r"([1][9][6-9][0-9])")) 

4. 获得该数据的用户约束：

uc.get_uc()

#### 使用教程

1. 安装依赖（或在conda虚拟环境下） 

pip install -r requirements.txt

2. 运行清洗（或在conda虚拟环境下） 

以hospital为例，运行命令：  python /example/hospital.py

可在/example/hospital.py中对infer_strategy、model_choice等参数进行修改。

在/example/hospital.py中model_save_path添加路径，即可保存模型的pkl文件。

