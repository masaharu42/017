# Fold03双目标力学反馈：代码与执行顺序

本次实现三项固定可微统计＋联合岭回归版本。**未启动真实代理拟合、诊断、采样或生成器训练。** 软件测试采用合成图和小型随机VAE/U-Net，不能证明科学效果。

## 软晶界是什么

将RGB由[-1,1]换算到[0,1]，计算：

`B = sigmoid((0.20 - max(R,G,B)) / 0.05)`。

它为每个像素提供连续分数：黑色约0.982；最大通道值0.20时为0.5；白色约0.0000001。意思是“像暗色晶界的程度”，不是经验证的真实晶界概率。暗色区域、线宽和亮度都可能影响结果，必须做扰动与硬指标核验。

连续分数使误差能向像素回传梯度；硬阈值/连通域不能直接代替这条可微路径。没有新CNN在这里训练。

## 实现对应关系

| 文件 | 功能 |
|---|---|
| `physics/soft_mechanics_features.py` | 版本化固定特征：B均值、B空间变化密度、内部32px水平/垂直S2；完整K组均值 |
| `physics/linear_mechanics.py` | 折内标准化、嵌套留一调岭系数、逐合金删除检查、冻结双输出代理 |
| `ebsd_diffusion/mechanics_groups.py` | 同合金连续K窗采样；组内共享时间步、扰动和条件丢弃 |
| `ebsd_diffusion/mechanics_runtime.py` | 标签掩码、等权标准化Huber、证据核验、FP32解码反传、分目标梯度校准 |
| `scripts/validate_mechanics_m0.py` | 原图特征提取、代理验收、完整嵌套置换、自由生成域移/K稳定性检查 |
| `scripts/probe_mechanics_inputs.py` | 固定真实窗的重建/扰动/单步去噪、偏差方向、干净图输入敏感性 |
| `scripts/probe_mechanics_gradients.py` | 强制零更新运行梯度探针 |
| `scripts/run_mechanics_pair.py` | 显式draft/prepare/calibrate/train-c0/train-c1，各阶段独立调用 |
| `train_diffusion.py` | 可选力学训练路径；没有`train_diffusion.mechanics`配置时保留原采样/损失路径 |

旧`MechanicsSurrogate`补齐forward供兼容调用；正式新方案使用`LinearMechanics`，不启用旧9目标MLP。

## 以后接CNN的接口

特征模块约定：`forward(BCHW RGB) -> [B, F]`，暴露版本、特征名和设置；当前没有可学习参数。`build_feature_extractor`只接受已实现的固定特征版本，未知`cnn`类型会明确报错。

后续CNN需另加实现、权重及哈希，重新拟合标准化和回归、重跑合金隔离验证及全部适用性门。当前三特征代理不能直接消费CNN特征。

## 上传/复用

准备22张原始RGB图、`data/manifest.csv`、`data/compositions.csv`、合金级与试次级力学标签及定义。内部默认排除ID03，终点仍读`endpoint_strain_pct`并标DIC终点代理。已确认损坏试次应在分析表中分目标重新汇总；可用`*_valid`列标注不可用合金目标。

S0使用约定纯MSE65k的raw U-Net＋对应成分编码器、标准化、scheduler、latent_scaling和匹配VAE。Mac的65k目录没有两项权重实体，要在原服务器核验复用。不能用EMA U-Net、结构统计组或不同VAE静默替换。

代码可通过版本库或文件包同步；服务器已有且哈希一致的数据/权重不必重复传。相对路径按项目根目录解析。所有新结果使用全新输出目录，脚本拒绝覆盖非空目录。

## 阶段一：真实图特征和代理（生成器不更新）

在项目根目录、具备项目依赖的Python环境运行。先按服务器实际路径复制/修改`configs/mechanics_m0.yaml`，保留ID03隔离：

```bash
python -m scripts.validate_mechanics_m0 extract --config configs/mechanics_m0.yaml --output output/mechanics_m0/extraction
python -m scripts.validate_mechanics_m0 fit --extraction output/mechanics_m0/extraction --output output/mechanics_m0/proxy --permutations 1000
```

提取时每合金32个固定随机窗口，另存最多两个不重叠角落窗口用于敏感性比较；每合金一行均值用于拟合，窗口不当独立合金。图像、标签、成分、特征表及位置池有哈希记录。

代理分别按目标检查：嵌套折外RMSE相对仅成分降低≥15%、至少ceil(12N/21)个合金绝对误差降低、删任一合金后增益不反转。1000次置换重跑完整嵌套，是无条件配对敏感性检查，不是严格条件独立检验。目标缺失不填虚构值；有效标签少于5时禁用该目标，不阻断另一个目标的独立验收。未拟合的头只保存禁用占位结构，禁止激活，不作为性能预测输出；小样本是否科学可用仍须审阅。

输出`proxy_model.json`、`proxy_validation.json`、`m0_decision.template.json`。**模板status为pending_review，不是训练放行文件。** `approved_targets`只记录代理预测门，不代表整个M0通过。

## 阶段二：适用范围与时间段（仍不更新生成器）

复制`configs/mechanics_pair.template.yaml`为`configs/mechanics_pair.local.yaml`，并将`configs/sample_mechanics_s0.template.yaml`复制为`configs/sample_mechanics_s0.local.yaml`，按服务器位置核对路径。后者必须指向65k raw权重及匹配VAE；模板的VAE路径来自回收的65k元数据。配对模板保存了已约定的U-Net/编码器哈希。

先只生成诊断用草稿配置；不会加载模型或开训：

```bash
python -m scripts.run_mechanics_pair --config configs/mechanics_pair.local.yaml --phase draft
```

终端会返回草稿配置目录。用该目录的`C1.yaml`运行真实窗口诊断：

```text
python -m scripts.probe_mechanics_inputs --config 草稿目录/C1.yaml --extraction output/mechanics_m0/extraction --output output/mechanics_m0/input_diagnostics
```

默认21合金×4窗口×3噪声种子，每个t为252个组合；t为100/200/300/450/600。组合不是独立样本。输出特征/预测的有符号偏移、重建基线、候选可靠t、亮度/线宽/模糊扰动及干净图像素读出梯度。VAE后验均值用于隔离加噪敏感性；U-Net精度沿用配置，解码/特征/代理FP32。

这份时间段结果明确保留`trajectory_review_pending`，扰动结果保留`hard_feature_review_pending`。真实干净图读出敏感性不是t=0的U-Net梯度。必须结合已有`diagnose_structure_trajectory.py`的可复现轨迹与独立硬组织读数审阅，不能由单步稳定自动推出覆盖布局阶段。当前轨迹复现问题未解决时，这条门不能凭空填passed。

自由生成域移检查需要一个CSV索引，列为`alloy_id,image_path,seed`；每个训练合金至少20个固定种子。路径相对索引所在目录或使用绝对路径。生成图必须来自同一S0及锁定采样协议，先核实其生成报告：

```bash
python -m scripts.validate_mechanics_m0 domain --extraction output/mechanics_m0/extraction --generated-index output/mechanics_m0/s0_generated_index.csv --output output/mechanics_m0/domain_k4 --k 4
```

K4不稳定时按预登记规则另运行K8，换全新输出目录。域移看最近邻与收缩协方差距离，总体和逐合金均须90%组在经验门内。窗口方差同时报告真实/生成两侧；两个不重叠窗口与随机窗口的偏差单列，仍需科学审阅。域外分数只作失配诊断。

## 阶段三：锁定适用性决策，准备配对实验

各门通过后另存人工审阅的`m0_decision.json`，不要改写原诊断结果。字段以模板为准：

- 绑定`proxy_sha256`、来源权重/配置`source_sha256`、VAE文件哈希；
- 锁定真实通过的`active_targets`、`group_size`、`max_t`；
- `window_stability`、`perturbations`、`domain`、`timestep_and_trajectory`各引用一份审阅结论JSON，记录其文件哈希；结论自身及决策中的status都必须是passed；
- 审阅结论需链接原始诊断和原图/轨迹证据，不得只改模板状态制造通过。

程序验证决策、证据、标签、图像及来源哈希，无法自动替代科学审阅。无目标通过、持续域外、时间段够不到布局或轨迹不可信时停止，不开两臂。

```bash
python -m scripts.run_mechanics_pair --config configs/mechanics_pair.local.yaml --phase prepare
```

prepare只加载核验与写配置，不更新参数。C0/C1共用同一采样、优化器、数据、学习率和所有力学设置，只允许arm与输出目录不同。

## 阶段四：零更新校准，再分别续训

```bash
python -m scripts.run_mechanics_pair --config configs/mechanics_pair.local.yaml --phase calibrate
```

默认200批，只做autograd.grad，不调用optimizer.step，不导出final权重。按目标分别记录梯度、二者夹角、与MSE夹角及U-Net/编码器范数。实际获准目标合并后标定lambda，使起点典型梯度比约15%，并绑定相同GPU/协议/代理/决策。

无足够活跃批、非有限梯度或明显相消导致虚大lambda时拒绝导出校准。两目标获准时各0.5；缺失目标不会使另一项自动翻倍。训练组内共享t/成分扰动/dropout，噪声独立；完整组门控，最多解码8窗。冻结VAE/代理，误差可沿输入穿过它们。

校准审阅完成后，两个阶段分别显式调用：

```bash
python -m scripts.run_mechanics_pair --config configs/mechanics_pair.local.yaml --phase train-c0
python -m scripts.run_mechanics_pair --config configs/mechanics_pair.local.yaml --phase train-c1
```

各从同一S0起点，用全新优化器续训5k；C1不从C0末尾接着训。准备后修改输入/配置/源码会被哈希检查拒绝。默认路径未设置mechanics时旧实验不受影响。

## 评价与当前边界

训练导出raw权重与力学元数据，记录损失/有效组数/像素越界/分目标及模块梯度和配对输入指纹。使用既有采样/独立硬指标流程，以相同20种子比较S0/C0/C1。不能用反馈代理同时充当唯一裁判；还要检查合金特异性、共同变细、多样性和复制风险。

本轮没有新增自动发表/成功结论功能，也没有替代已有评价脚本；ID03后续输出封存，训练内结论锁定后探索性报告。两目标同训无法分离各自的因果贡献。

## 软件验证

`tests/test_mechanics_feedback.py`覆盖连续软分数、空间敏感性、冻结解码反传、成分支路detach、标准化等权/缺失标签、组采样/共享随机输入、目标梯度冲突、折外标签隔离、M0不自动放行，以及真实小型VAE/U-Net的校准＋C0/C1合成续训。另运行原结构统计和Min-SNR流程回归。

这些合成测试中的passed证据仅存在于临时fixture，不能用于真实M0。
