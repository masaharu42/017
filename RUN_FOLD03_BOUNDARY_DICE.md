# Fold03 raw50k＋10k：局部晶界 Dice 筛查

只训练 C；读取已完成的 A_mse，不重训A。A和C从同一raw50k与原成分编码器开始，各新增10k；不加局部五维、Min-SNR、旧GB/edge损失或力学反馈。A与C训练如跨GPU，仅作为筛查对照；不声称训练逐批逐位一致。固定256、batch32、FP16、LR3e-5、warmup500、cosine，其他设置从已验证A配置继承。评估固定ID05/07/20、两种子、DDIM50、CFG1.8、raw；ID03不用于筛选。

训练中当t≤300时，按批内顺序最多8张，估计干净latent并用冻结的VAE解码；预测RGB映射到[0,1]后用sigmoid((0.2-maxRGB)/0.05)形成软晶界图，目标用真实小图maxRGB<0.2的硬掩码，无膨胀。对每图计算soft Dice并求平均。只要没有符合条件的样本，该批晶界loss为零。该目标是RGB黑线代理，无法保证晶粒闭合或物理真实。使用像素级配对的去噪重建监督，自由生成并不对应固定真实窗口。

## 服务器准备

在服务器项目根目录同步后，保留已完成的`08_训练与生成结果/06_Fold03_raw50k_MSE_MinSNR5_10k_运行/A_mse`及其manifest、配置、源码快照、数据和来源权重。无卡模式可以同步代码；prepare需要足够内存加载检查模型，不需要CUDA。校准、训练与采样需要有卡。当前任务未做真实CUDA测试；先执行校准，观察日志/显存。校准可能比普通训练占更多显存，如果不兼容则停止并反馈日志，不能降低batch或关闭检查后声称严格同条件。

```bash
cp configs/fold03_boundary_dice_10k.yaml configs/fold03_boundary_dice_10k.local.yaml
python -u -m scripts.run_boundary_pilot --config configs/fold03_boundary_dice_10k.local.yaml --phase prepare
```

默认路径为同一服务器项目中原06运行目录。若不同，修改本地YAML中baseline_run；output_root必须是全新目录。prepare核对A已完成、来源50000、原配置与权重和数据身份，冻结本次代码、配置及来源哈希。不覆盖原目录；校准开始后不要改配置或代码。

## 零更新校准与训练

在有CUDA的同一服务器/相同GPU上分别执行：

```bash
nohup python -u -m scripts.run_boundary_pilot --config configs/fold03_boundary_dice_10k.local.yaml --phase calibrate > boundary_calibrate_launcher.log 2>&1 < /dev/null &
tail -f boundary_calibrate_launcher.log
```

校准对8批分别测整批MSE和门控晶界loss的U-Net裁剪前梯度范数，不调用optimizer.step；取每批MSE/晶界梯度范数比值的中位数×0.1作为固定λ。非有限/零梯度或有效批次不足则停止。报告`calibration/boundary_calibration.json`中记录每批数据与λ；10%是初始筛查量级，不保证训练全程的梯度比例。梯度余弦未测量，10%不代表任务方向一致。校准结果和训练配置/设备绑定。校准不会创建A，也不会更新原始权重。

确认校准末尾`status: calibrated`后，再运行：

```bash
nohup python -u -m scripts.run_boundary_pilot --config configs/fold03_boundary_dice_10k.local.yaml --phase train > boundary_train_launcher.log 2>&1 < /dev/null &
tail -f boundary_train_launcher.log
```

结果写到新目录`C_boundary`，每2000步checkpoint，日志中有boundary_dice_loss、selected、lambda；最后导出raw。后台nohup可应对SSH断开，无法防止容器关闭、磁盘耗尽或进程报错。部分结果不会自动重启/恢复/覆盖。旧A不同GPU训练产生的数值差异保留为限制。

训练完成后独立采样：

```bash
nohup python -u -m scripts.run_boundary_pilot --config configs/fold03_boundary_dice_10k.local.yaml --phase evaluate > boundary_evaluate_launcher.log 2>&1 < /dev/null &
tail -f boundary_evaluate_launcher.log
```

在当前GPU用A与C的raw权重重新采样各6张，初始latent和DDIM网格逐对核验；不会重训A。`evaluation/alloy_*/comparison_grid.png`为两组同种子对比，`real_windows.png`为真实窗口，`per_image_metrics.json`及`sampling_inputs.json`存逐图指标和采样记录。窗口可能重叠；两种子只作筛查，不能推断泛化或力学。请同时检查黑线厚度、碎点、颜色及组织形态，不能单凭Dice或区域数判成功。
