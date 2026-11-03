# 单尺度纯图像CNN训练顺序

代码只用RGB组织窗口预测峰值应力与DIC终点应变，不将成分输入CNN。成分仍用于后续扩散条件。网络为16/32/64通道卷积、GAP、64→20→12→2回归头；原生256窗口，暂只在线随机裁切。组内K=4预测取均值，两个目标在各训练折分别标准化后等权MSE。

## 服务器同步和输入

在服务器017项目根目录、原Python环境执行`git status --short`，保留本机更改。工作区无冲突时执行`git pull --ff-only origin main`，不要用reset丢弃服务器已有修改。同步成功后运行以下步骤。Git只同步代码；原图、标签、旧M0提取资料与权重继续使用服务器现有文件。

需要：`output/mechanics_m0/extraction/{extraction.json,alloy_features.csv}`、data中的manifest及原图、现有合金力学中位表。运行前核对模板中的这些路径。要求Python>=3.11和项目依赖。

```bash
cp -n configs/mechanics_cnn.yaml configs/mechanics_cnn.local.yaml
python -m scripts.train_mechanics_cnn --config configs/mechanics_cnn.local.yaml --phase prepare
```

`prepare`只核验和锁定输入、源图、代码和配置哈希，不训练。默认输出`output/mechanics_cnn_v1`必须为新目录。锁定后修改输入、代码或配置会拒绝继续；保留原运行另建版本。

## 先训练并验收CNN

```bash
mkdir -p logs
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
nohup python -u -m scripts.train_mechanics_cnn \
  --config configs/mechanics_cnn.local.yaml --phase cv \
  > logs/mechanics_cnn_v1.log 2>&1 &
```

```bash
tail -f logs/mechanics_cnn_v1.log
```

21个外层留一合金×3个固定种子。每个外层的20个训练合金再做4折内层分组，最多200轮，仅内层验证选择训练轮数，取内层最佳轮数中位数在全部20个合金重训，最后只预测留出的那个合金。默认每轮128个合金组、每批8个不同合金、每合金4窗；不是把大量增强窗口当独立合金。

共315次小网络拟合，CPU读取原图、GPU训练；耗时以服务器实测为准。固定全部种子，不能只报最好一轮。完成后生成`summary.json`：21个合金的折外预测、每个种子与集成RMSE/MAE/R²，以及折内训练均值基线。ID03不参与训练或选择。该命令不加载生成器，不会启动C0/C1。

也支持单折，例如：

```bash
python -u -m scripts.train_mechanics_cnn --config configs/mechanics_cnn.local.yaml \
  --phase fold --alloy-id 1 --seed 20261103
```

`cv`重启时仅跳过已完整且哈希匹配的折。中断的半成品折保留并报错，不静默覆盖；先回收日志和结果再决定如何恢复。全部折完成后可独立执行`--phase collect`汇总，已存在的汇总不覆盖。

将整个新输出目录回传复核。本版预登记的探索性预测初筛为：集成折外R²≥0.5、至少12/21合金的绝对误差低于训练折均值基线、每个种子折外R²均为正。这是操作性初筛，不是普适物理门，也不是旧“比成分改善15%”的改判。RMSE/MAE绝对量级、标签与结构敏感性仍需科学审阅，不能只凭初筛授权反馈。当前汇总不自动批准部署重训或扩散训练。旧M0-v2未通过结论不变。

## 再拟合用于部署的CNN

仅在独立审阅预测结果后，用明确的review文件执行`--phase refit --review 文件路径`。review格式：

```json
{
  "schema": "image_cnn_prediction_review_v1",
  "status": "passed",
  "summary_sha256": "实际summary.json的SHA256",
  "active_targets": [true, true],
  "rationale": "实际预测验收依据及两个目标分别的判读"
}
```

这不是空白许可模板：不能照抄填成passed绕过科学审阅。某目标不可靠时不激活。refit用全部21个开发合金，按各外层选出的轮数中位数训练3个种子，生成权重、目标归一化和`refit/cnn_proxy.json`。评估指标仍取折外汇总，不能用最终21合金拟合误差代替泛化误差。

## 最后才准备扩散反馈

1. 冻结CNN，检查真实图、VAE重建、单步x0估计和S0自由生成图的域差、窗口组稳定性与合理结构/颜色敏感性。
2. 新CNN需独立的`image_cnn_feedback_decision_v1`决策，绑定CNN工件哈希、S0、VAE、active_targets、group_size、max_t以及window_stability/perturbations/domain/timestep_and_trajectory四项证据。旧线性M0决策不能授权CNN。
3. 200个零更新批次测力学单独对图像、潜变量、各U-Net块和条件编码器的梯度；标定λ。CNN/VAE参数与BN统计保持不变。
4. 证据通过后才准备C0/C1：同一S0、预算、种子、组采样、优化器与参数范围；C0普通MSE，C1加标准化Huber力学项。两目标标准化后等权。整个U-Net与条件编码器可更新；CNN和VAE冻结。
5. 当前反馈位置是训练时单步干净图估计，不是从纯噪声展开完整50步采样。最终自由生成与独立硬组织、复制、多样性评价仍需单独完成。

反馈接口现已支持CNN工件，旧的线性代理仍可用；没有真实CNN验收、域与梯度证据时不会因此自动开训。数据输入与证据哈希校验要求对应同一服务器项目路径。
