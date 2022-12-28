# Torch Traning Template

## Introduction

Hi~ 这是一个 `Pytorch` 的训练模板, 总体使用原则是: 根据一个配置文件, 在 `trainer` 对象中载入相应的各个模块, 并根据自定义的 `train_epoch()` &  `test_epoch()` 开始一次训练. 总之, 它具有以下优点:

1. 目录结构清晰, 文件归类合理, `toml` 配置文件易用
2. 封装简单且基础, 你很容易基于此模板定制自己的东西, 拓展性强
3. 附带一些小工具, 如自动选择 GPU 的 shell 脚本, 有额外的 `resnet1D` 等模型, 还有多个 `trainer` 案例
4. console 中具有进度条显示(可带颜色), 有 txt 日志记录, 有 tensorboard, 还有画混淆矩阵......

它简约又简单, 能快速上手, 希望可以帮到你!

## Structure of The Project!! 



## Requirements

- Python $\ge$ 3.8

- tqdm

- tensorboard

- toml

## A Quick Start



## TODO List

- [x] 基本结构
- [x] `tqdm` 集成 
- [ ] 提供 [Domain-Adversarial Training of Neural Networks, DANN](https://arxiv.org/abs/1505.07818) 示例
- [ ] 仿照 `keras` 提供 `train_step` & `test_step` 接口

