# 基于向量符号架构与双注意力机制的非配对图像翻译（DaVSAIT）

**Hongqian Chen**<sup>1,3</sup>、**Jiahui Xiong**<sup>1,3</sup>、**Xiaochuan Wang**<sup>1,3</sup>、**Hui Li**<sup>2,*</sup>

<sup>1</sup> 北京工商大学 计算机与人工智能学院<br>
<sup>2</sup> 北京联合大学 管理学院<br>
<sup>3</sup> 北京工商大学 食品安全大数据技术北京市重点实验室<br>
<sup>*</sup> 通讯作者：lihui@buu.edu.cn

[English](README.md) · [[预训练模型]](../../releases)

## 简介

针对源域与目标域语义差异较大时非配对图像翻译中的细节丢失和语义翻转问题，DaVSAIT 在 VSAIT（ECCV 2022）基础上为生成器引入**双注意力机制**：

- **多光谱通道注意力**（参考 FcaNet）：捕捉通道间依赖，增强纹理与边界细节；
- **自注意力**（参考 SAGAN，可学习系数 γ 初始为 0）：捕捉长距离空间依赖，提升全局一致性。

模型由三部分组成：双注意力生成器 G_att、超向量编码器 E（VGG19 + LSH）、双向可逆映射网络 R（两层 MLP）。

在 GTA→Cityscapes 任务上，KID 和 FID 比 VSAIT 分别降低 29.2% 和 15.6%，pxAcc / clsAcc / mIoU 均为最优。完整结果表见 [README.md](README.md)。

## 实验任务

| 任务 | λ | 训练轮数 |
|---|---|---|
| GTA → Cityscapes | 10 | 20 |
| Google Maps ↔ Aerial Photo | 5 | 400 |
| Cityscapes → CBCL | 10 | 20 |

## 安装、训练、测试

环境：Python 3.8.13、PyTorch 1.10.0（CUDA 11.3）、PyTorch-Lightning 1.5.9，单卡 RTX 3090。
具体命令见 [README.md](README.md) 或 `scripts/` 文件夹；预训练权重在 [Releases](../../releases) 页面下载，放到 `./pretrained/`。

## 致谢与许可证

本工作得到国家自然科学基金（No. 62307001）资助。代码基于 VSAIT（Theiss 等，ECCV 2022），沿用其 CC-BY-NC 4.0 许可证（仅限非商业用途）。
