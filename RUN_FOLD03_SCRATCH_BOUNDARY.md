# Fold03 从零训练晶界 Dice：大显存筛查

本实验从零初始化 U-Net 和成分编码器，复用 A 既有 VAE、原数据、噪声日程和 latent scaling；不加载扩散权重或优化器状态。50,000步、batch32、累积1、FP16、LR 1e-4、warmup1000、cosine，关闭梯度检查点，非 EMA raw 作为生成权重。目标函数为 epsilon MSE 加晶界 Dice：仅t≤300、每批最多8张，复用原实验定义。损失权重从新初始化模型做8批零更新梯度校准，固定为梯度范数中位比×0.1。条件随机丢弃0.15；不使用五维或力学反馈。

这是独立实验，不能宣称关闭检查点后GPU位级相同或性能必然相同。batch32维持历史每步样本数和更新次数；batch64会改变训练动力学，此协议不擅自增加batch。真实GPU上尚未验证吞吐或显存，不能保证任何时长；预计剩余时间以实测benchmark为准。

## 步骤

服务器已有已完成的06 A基线、原VAE/原图及数据；在项目根目录，准备本地配置。prepare核对旧基线和来源，检查新建DDPM调度的alpha与旧调度完全一致，冻结输入和代码哈希。output_root必须是全新目录，校准后不要改变配置、源码或数据。

```bash
cp -n configs/fold03_scratch_boundary_50k.yaml configs/fold03_scratch_boundary_50k.local.yaml
python -u -m scripts.run_boundary_scratch --config configs/fold03_scratch_boundary_50k.local.yaml --phase prepare
```

进入有CUDA服务器后按顺序运行：

```bash
nohup python -u -m scripts.run_boundary_scratch --config configs/fold03_scratch_boundary_50k.local.yaml --phase calibrate > scratch_calibrate.log 2>&1 < /dev/null &
tail -f scratch_calibrate.log
```

calibrate 完成并报告 calibrated 后：

```bash
nohup python -u -m scripts.run_boundary_scratch --config configs/fold03_scratch_boundary_50k.local.yaml --phase benchmark > scratch_benchmark.log 2>&1 < /dev/null &
tail -f scratch_benchmark.log
```

benchmark运行独立30步临时更新，前5步预热；不导出模型，不污染正式模型初始化。输出benchmark/benchmark.json包含每秒图像数、每步秒数和峰值显存，按seconds_per_step×50000粗估，不含准备、保存和评估。测速后从相同种子重新建立模型，不接续测速权重。

benchmark 完成并报告 benchmarked 后：

```bash
nohup python -u -m scripts.run_boundary_scratch --config configs/fold03_scratch_boundary_50k.local.yaml --phase train > scratch_train.log 2>&1 < /dev/null &
tail -f scratch_train.log
```

开始前检查校准/配置/基线/代码哈希、同GPU、空闲显存超过测速峰值加1GiB、磁盘空余至少12GiB。12GiB是保守启动门槛，不是运行空间保证。任何失败或中断的目录都保留，不会自动重试、恢复或覆盖。50k每1万步保存一次，原始对照与结果不删除。nohup不防止服务器关机、磁盘满或程序崩溃。

训练完成后生成原A raw50k与新模型的共享噪声对照（ID05/07/20及先前已看过的探索性ID03，两种子，每组每合金2张；无EMA）：

```bash
nohup python -u -m scripts.evaluate_boundary_scratch --config configs/fold03_scratch_boundary_50k.local.yaml > scratch_evaluation.log 2>&1 < /dev/null &
tail -f scratch_evaluation.log
```

结果位于evaluation_scratch，每个alloy目录包含comparison_grid.png、real_windows.png和原图；根目录为逐图指标、噪声身份和哈希。ID03此前已经用于开发诊断，不是严格盲测。自动评价不会跟在训练后运行；先检查训练成功和磁盘情况。服务器运行报错时保留日志与目录。
