# Torch Training Template

## 1. Introduction

Hi~ 这是一个 `Pytorch` 的训练模板, 总体使用原则是: 根据指定的 `toml` 配置文件载入核心训练逻辑即 `trainer` 对象, 并调用构造函数 `__init__()` 从配置文件中载入相应的各个模块及参数(模型, 优化器, 数据集等); 最后调用 `trainer` 中的 `train()` 方法开始一次训练, 期间也会有日志输出. 简而言之, 它具有以下优点:

1. 目录结构清晰, 文件归类合理, `toml` 配置文件比 `json` 易读易用
2. 封装简单且基础, 你很容易基于此模板灵活定制自己的东西
3. 附带一些小工具, 如自动选择 GPU 的 shell 脚本, 有额外的 `resnet1D` 等模型, 还有多个 `trainer` 小例子
4. 较完整的 typehint 支持
5. console 中具有彩色进度条显示(基于 `rich`), 有 txt 日志记录, 有 tensorboard 示例, 还有画混淆矩阵......

它简约又灵活, 能快速上手, 还有 3 个 trainer 例子希望可以帮到你! 使用前读完此文件即可避免大部分问题.
> trainer 例子包含: 一个最普通的训练逻辑, 一个 DANN (Domain-Adversarial Training of Neural Networks) 的训练实现, 还有一个 MCD (Maximum Classifier Discrepancy for Unsupervised Domain Adaptation) 的训练实现

## 2. Structure of The Project!
主要由以下文件或文件夹组成:
- set_cuda.sh (文件)
  
  一个自动设置环境变量 `CUDA_VISIBLE_DEVICES` 的脚本, 在多 GPU 设备上按剩余显存排序进行选择, 一般在开启一个新 shell 窗口后就执行一次. 可指定 GPU 数量, 用法如下:
  ```shell
  . ./set_cuda.sh 2  # 开头的 '.' 不可忽略!!! '.' 可以换为 source
  . ./set_cuda.sh  # 默认 1 块 GPU
  ```

- main.py (文件)

  程序最外层入口, 在随机数种子等设置过后, 把 `toml` 配置文件读取为一个 Python 字典即 `info`, 并根此载入 `trainer` 对象. 开启一次实验只需要: 
  ```shell
  python main.py -c configs/mnist.toml
  ```

- configs (文件夹)

  用于存放某一次实验的 `toml` 配置文件. 它主要包含了这次实验要读取的数据集 `dataset`, 核心训练逻辑 `trainer`, 重要超参数和学习率策略 `lr_scheduler` 等等, 可以自己按需拓展.
  设置绝大多数会被具体 `trainer` 及其父类的 `__init__()` 根据 `main.py` 产生的 `info` 字典载入, 自己拓展时可以参考此处代码. 
  现已支持自动载入多个数据集, 有且仅有一个训练集(名称为 dataloader_train), 可以有零个或多个测试集(名称须包含 dataloader_test).

- datasets (Python package)

  存放标准的 pytorch 格式的数据集实现, 应该返回 `torch.Tensor` 对象. 
  
- models (Python package)

  放置一些继承自 `torch.nn.Module` 的神经网络模型. 至于是完整的端到端模型, 还是计算流程中的一个小模块, 随你便咯~ 它们怎么被使用是在 `trainer` 中被你定义的.

- saved (文件夹)

  存放训练的中间输出, 文件夹名字由 `toml` 文件控制. 输出包括: 
  1. 针对 `trainer.test_epoch()` 中测试结果的混淆矩阵图片, 需要在 `toml` 中设置 `plot_confusion` 为 `true`; 
  2. 训练日志. 包含此次实验所用的 `toml` 文件, tensorboard 日志与 `txt` 日志;
  3. 训练过程中的模型权重, 和最佳表现的权重. 

- trainers (Python package)

  此处为训练的核心逻辑. 新建的 `trainer` 类应该继承自 `_Trainer_Base`, 因为其中有可复用基本变量与方法的设置. 新的 `trainer` 类可以在 `__init__()` 中载入一些基本变量, **必须**先自定义这几个方法: 
  1. _prepare_dataloaders, 根据 `toml` 载入数据集;
  2. _prepare_models, 根据 `toml` 载入需要用的模型, 可以根据 `resume` 变量传入的路径恢复模型;
  3. _prepare_opt, 定义模型对应的优化器与学习率策略;
  4. _reset_grad, 为方便, 把若干个优化器的清空梯度包装于此.
  
  接下来就需要完成真正的核心部分: 训练逻辑 `train_epoch()` 与测试逻辑 `test_epoch()`. 它们迭代完整个 epoch 后应该返回由指标 (metrics) 构成的字典, 此处亦可在 value 处返回一个 `(值, 颜色字符串)` 元组, 使得在 console 中把这个指标变彩色. 具体颜色的定义在 `_Trainer_Base` 的开头. 
  
  如果需要画混淆矩阵, 可以用 `plot_confusion` 包装某一个 step 比如 `test_epoch()`, 其内还需要把真实标签与预测结果保存到公共变量 `self._y_true, self._y_pred` 中(此部分不写相当于不画), 用以画混淆矩阵图. 装饰器 `plot_confusion` 会自动解析这两个公共变量去画混淆矩阵. 

  修改写入日志的行为或按 epoch 修改学习率则需要覆盖父类的 `train()` 方法.

  接下来还有些辅助函数可以使用: 
  
  1. _epoch_end, 一个 epoch 的训练与测试结束后的行为, 暂时为空.
  2. _train_end, 全部流程结束后的行为, 子类覆盖时别忘了在方法末尾调用 `super()._train_end()`, 因为要关闭 tensorboard 句柄, 否则最后一个记录点很可能无法被写入.
  3. plot_confusion, 一个装饰器, 用于方便地画混淆矩阵, 用法见代码和前面的描述.

- utils (文件夹)

  放一些杂七杂八的小工具, 比如制作 MNIST-M 数据集的脚本, 按通道获取数据集均值与标准差的函数等.


## 3. Requirements

- Python $\ge$ 3.9 (Type Hints)

- pytorch

- rich

- tensorboard

- toml

## 4. A Quick Start
1. clone 此项目或直接下载压缩包并解压; 删除 `.git` 文件夹, 可删除 `saved` 文件夹
```shell
git clone https://github.com/LEON-REIN/TorchTrainingTemplate.git --depth=1
```

2. 在多 GPU 机器上手动或调用脚本设置环境变量 `CUDA_VISIBLE_DEVICES`, 单 GPU 机器可不管
```shell
# 开头的 '.' 不可忽略!!! '.' 可以换为 source; 
. ./set_cuda.sh 2  # 指定 2 块剩余显存最大的 GPU
. ./set_cuda.sh  # 默认指定一块
```

3. 切换工作目录, 激活合适的 Python 环境

4. 修改 `'datasets/mnist_raw.py'` 第 5 行的 `path` 为你专存放数据集的文件夹, 也可直接改为 `'./'`; 将第 13, 14 行的 `download` 设为 `true`

5. 执行一次基本的 MNIST 分类训练, 无 MNIST 数据文件夹则会自动下载到上一步中指定的目录
```shell
python main.py -c configs/mnist.toml
```

6. 观察 console 的输出, 也可用 ` tensorboard --logdir "saved"` 命令看实时的指标变化, 也可喝一杯茶~

## 5. Notes

- 如果机器性能较差, 请将 `toml` 配置中传给 `dataloader` 的 `pin_memory` 设为 `false`, 并调小 `num_workers` 甚至为 0

- 在含有 `__init__.py` 的文件夹中建立新文件或模块时, 需要在 `__init__.py` 中 `import` 一下该模块, 否则会找不到

- 文件下不动可以使用 [fastgithub](https://github.com/dotnetcore/fastgithub)

- 默认会读取 `toml` 中字段**等于** `dataloader_train` 的数据集和**包含** `dataloader_test` 的多个数据集(之后都以字典序被处理), 分别作为变量 `self.dataloader_train` 和 `self.dataloader_test_dict'; 默认会读取其中字段**等于** `model` 的部分作为模型; 同样地, `lr_scheduler` 字段也是默认被读取. 可以覆盖基类以自定义这些行为

- 模型恢复可以写在 `toml` 的 `resume` 字段下, 如果命令行也传入会覆盖 `toml` 中的设置, 并且在相应的 `log` 文件夹中的 `toml` 的首行添加对应信息

