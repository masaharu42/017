# 017：物理感知 EBSD 欧拉角扩散项目

本项目从 01 项目的可复现代码、原始 RGB IPF+GB 数据、成分表和已训练权重建立。

## 当前已包含

- 01 项目的 VAE 训练与权重；
- 01 项目的扩散训练与权重；
- 22 个 RGB IPF+GB 原始图；
- 成分、热处理和已有 EBSD 统计表；
- 物理描述符与力学反馈模块的接口代码。

### 权重清单

- VAE：`08_训练与生成结果/01_VAE模型/final/diffusion_pytorch_model.safetensors`
- 扩散 U-Net：`08_训练与生成结果/02_扩散模型/final/unet/diffusion_pytorch_model.safetensors`
- 条件编码器：`08_训练与生成结果/02_扩散模型/final/condition_encoder/pytorch_model.bin`

这些是 01 项目的已训练 RGB 基线权重，使用 Git LFS 管理。

## 重要数据状态

当前 01 项目中没有逐像素原始 Euler 角文件，只有已经导出的 IPF+GB 图像。因此 Euler 编码版本还不能直接训练。收到 `.ang`、`.ctf`、`.h5` 或逐像素 Euler CSV 后，放入 `data/raw_orientation/`，再运行转换脚本。

## 设计目标

1. 先保留 01 项目的稳定基线：RGB VAE、普通 epsilon MSE、CFG≈1.8；
2. 新增 Euler/取向表示，不同时改变所有训练因素；
3. 用可微晶粒/边界统计约束生成结果；
4. 用冻结的力学代理模型增加应力应变反馈；
5. 每个新增 loss 都先做梯度探针和 A/B 对照。

力学反馈模块只有在提供真实应力应变标签并训练/验收代理模型后才能启用，当前不会伪造力学监督。

## 当前边界

017 当前可以复现 01 项目的 RGB VAE+扩散流程；`physics/` 中的 Euler 编码、可微描述符和力学反馈是后续接入接口。由于 01 项目没有逐像素 `.ang`、`.ctf`、`.h5` 或 Euler CSV，也没有可验收的应力应变标签与代理模型权重，当前不能宣称 Euler 训练或力学反馈训练已经完成。

获得原始取向数据后，先生成 Euler 表示并做小规模 VAE 重建验收；获得力学标签后，先单独训练并验证代理模型，再接入扩散 loss。两者都不会覆盖现有 RGB 基线。
