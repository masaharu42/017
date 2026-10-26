# Fold03 raw50k：MSE / Min-SNR γ5 两臂10k实验

本协议是独立的新训练实验，不覆盖Stage A的A20k/B2/C2主表，也不改旧5k协议。
Mac只改代码、做软件测试和通过SSH审阅；**训练和本次统一评价都在AutoDL服务器完成，不经过Windows。**

## 跑什么

两臂独立加载同一A raw50k U-Net及同checkpoint raw编码器。`A_mse`目录对应M组普通epsilon MSE；`B_minsnr5`对应W组标准Min-SNR γ5。名称不代表历史B2/C2。

每臂新增10000次优化更新（来源50k，有效总步数各60k），峰值LR `3e-5`，warmup500后cosine到0；batch32、累积1、patch256、FP16、gradient checkpointing开启，workers4，seed20260721。两臂串行在同一单GPU/环境运行。推荐5090 32GB；服务器环境必须支持该GPU，并以实际运行检查显存/内核兼容性。

继续训练U-Net和编码器；冻结VAE、归一化和latent_scaling；新建AdamW，不加载旧optimizer/LR/RNG/EMA状态。weight decay=1e-4，梯度裁剪1.0，jitter0.05、condition dropout0.15，无新增增强。两臂GB/edge权重均为0，原1000步cosine forward日程不变。每2000步保存、每25步记录，主判只用10000步末尾raw及对应编码器。

本预算是一次有限的配方筛查，不是已验证最优值。加权不直接监督晶界拓扑；阴性不自动触发加步数、改γ或叠loss，也不否定从头训练的Min-SNR。

## 服务器准备

VS Code Remote SSH进入**服务器项目根目录**，激活已有兼容环境。先检查工作区；有冲突改动时核对后再同步，不使用强制覆盖。

```bash
git status --short
git pull --ff-only origin main
nvidia-smi
python -c "import torch,diffusers,accelerate; print('torch',torch.__version__,'CUDA',torch.version.cuda,'diffusers',diffusers.__version__,'accelerate',accelerate.__version__); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
cp configs/fold03_raw50k_minsnr_10k.yaml configs/fold03_raw50k_minsnr_10k.local.yaml
```

只编辑该`.local.yaml`的路径设置：

- `data_root: null`表示数据/权重都按当前checkout解析。如果数据归档在其他目录，填服务器实际归档根目录，例如`/root/autodl-tmp/017_data`。此目录应包含`data/`及来源配置引用的`01_基线留一验证_同一配置/`、`02_256空间匹配实验/`等相对层级。`data/manifest.csv`的原图路径必须能从manifest所在目录或其上一级解析。
- `source_sampling_config`、`historical_hparams`本身仍相对代码checkout解析；来源配置内的模型路径与历史hparams内的数据路径相对`data_root`解析。绝对路径原样保留。
- 若服务器目录结构不同，可另存`configs/sample_a_raw50k_server.local.yaml`，将来源配置中的权重/VAE/配置文件路径写为核实过的绝对路径，再把协议的`source_sampling_config`指向它。不得凭目录名把50k编码器换成20k。
- `output_root`必须是全新实验目录；默认以`_运行`结尾，已被Git忽略。正式prepare后不再改配置/代码/来源文件；改动会被拒绝，应保留旧结果并换新实验目录重新准备。

Git不包含完整训练数据和50k权重，必须先放到服务器。新协议锁定了已归档raw U-Net、编码器、共用VAE的二进制SHA256，以及来源步数和latent_scaling；不匹配就停止，不改哈希绕过。

## 执行顺序

以下每条分别执行，上一步成功后才执行下一条；不会自动启动后续阶段。

```bash
python -m scripts.run_minsnr_ablation --config configs/fold03_raw50k_minsnr_10k.local.yaml --phase prepare
```

`prepare`只做CPU加载、路径/数据/权重一致性检查，生成两臂配置、源码快照、Git身份和输入哈希，不训练。核对输出中的训练合金为排除ID03的21个、来源50000步、batch32/256/FP16及新增10000步。

确认后，在服务器的tmux会话或其他后台终端中执行，避免VS Code连接断开影响作业：

```bash
python -m scripts.run_minsnr_ablation --config configs/fold03_raw50k_minsnr_10k.local.yaml --phase train
```

它依次完成`A_mse`与`B_minsnr5`，保存各自完整stdout/stderr到`console_logs/`。非有限loss/梯度、AMP跳过更新或输入抽查不匹配会停止。部分失败不自动续训/覆盖；已有中间checkpoint不是“完全相同随机流恢复”的承诺。

训练完成后，同一服务器执行训练内统一评价：

```bash
python -m scripts.run_minsnr_ablation --config configs/fold03_raw50k_minsnr_10k.local.yaml --phase evaluate
```

生成起点raw50k/M末步/W末步三列，ID05/07/20各五种子，共45张。固定256、DDIM50、CFG1.8、FP16，真实图不进入生成latent；三列实际初始latent哈希和DDIM网格必须一致。未检测到CUDA时拒绝本协议的评价，不静默换成CPU/FP32。输出到`evaluation/`，不会混入原Stage A主表。

先审阅训练内结果，把结论写入一个UTF-8文字文件，例如实验目录中的`monitor_review.txt`。这是锁定已经看过的训练内结论，不是根据ID03选法。随后才执行一次ID03探索性报告：

```bash
# 将下列两处运行目录替换为local.yaml实际output_root。
RUN_DIR='08_训练与生成结果/06_Fold03_raw50k_MSE_MinSNR5_10k_运行'
python -m scripts.run_minsnr_ablation --config configs/fold03_raw50k_minsnr_10k.local.yaml --phase evaluate-heldout --review-note "$RUN_DIR/monitor_review.txt"
```

ID03三列×五种子共15张，输出到`evaluation_heldout/`；不会覆盖训练内评价。ID03已经参与历史开发诊断，只能称探索性留出报告，不能称盲测或全链严格留一。

## 看什么结果

- `experiment_manifest.json`、`configs/protocol.yaml`、两臂配置、`source_snapshot/`：代码/数据/权重/协议身份。Git状态与源码哈希一同记录；没有Git时如实记null。
- 两臂`training_metrics.csv`：实际LR、未加权MSE、区间平均加权/未加权误差、平均权重、裁剪前梯度范数和裁剪比例。`timestep_metrics.csv`：每个日志区间10个t分箱的样本数及加权/未加权误差；空箱留空，不能当误差为0。`training_diagnostics.json`计入全部成功更新，不是仅抽查日志步。
- `input_fingerprints.csv`和`matched_input_verification.json`：前5批及每25步的图像/latent/噪声/t/条件扰动抽查，不冒充全步穷尽证明。
- `final/unet_raw`与`final/condition_encoder`：正式末步权重。`final/unet`仍沿用历史EMA导出接口，**不进入本轮评价**。
- 评价目录中的`comparison_grid.png`、`summary.csv/json`、`per_image_metrics.json`、`sampling_inputs.json`和`run_metadata.json`：全部图片、真实窗坐标、实际噪声身份、模型/PNG哈希及执行范围。
- `paired_comparison.csv/json`：按生成seed配对，以同一合金五窗中位数为参考的区域数/GB/edge绝对对数偏差。`delta_mse_minus_minsnr>0`表示W的该代理误差较小，不是自动验收通过。无有效区域的图单独标记，不把旧指标中的哨兵count=1当正常区域统计。

主要看W对M：ID05/07的分区、渐变和颜色能否共同改善；ID20不能以退化为代价。结合全部五种子、面积分布与色彩/空间统计，不能靠黑线变粗或碎片增多刷区域数。两臂共同改善不单独归功于Min-SNR；加权loss较低不等于图像更好。五个真实窗口可能重叠，RGB连通区域不是物理晶粒数，色相JS距离不是取向分布；单次训练种子不提供训练重复方差。

## 在Mac审阅，不经过Windows

有可用SSH时，Mac助手可直接读取服务器CSV/日志，取必要PNG到Mac查看；权重和数据留在服务器。仅打开VS Code远程窗口，不意味着另一个Mac会话已取得SSH连接，需先确认连接别名和服务器项目路径。

也可在服务器打包小结果归档（不重新训练/采样，不含权重/检查点/原始数据）：

```bash
python -m scripts.pack_minsnr_review --run-dir "$RUN_DIR"
```

ID03报告前生成`review_bundle_monitor.tar.gz`，之后生成`review_bundle_with_id03.tar.gz`；同名不覆盖。包内`REVIEW_FILES_SHA256.json`列出文件哈希/大小，包旁有`.sha256`。需要本地保存时只下载这个包，告诉Mac助手路径即可。旧5k协议和PRO6000大batch协议均不属于本轮命令。
