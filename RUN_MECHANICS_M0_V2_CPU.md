# M0-v2：CPU审计与硬形貌对照

本轮是看过v1阴性结果后的第二轮开发。代码提供CPU分析，不修改标签、不删除ID21、不使用ID03选择方案，**不会训练CNN、不会加载生成器、不会放行C0/C1**。

## 同步与输入

将`mechanics_m0_v2_cpu_sync_20261102.tar.gz`上传服务器。解包后按包内相对路径同步到017项目根目录；同名文件先保留备份，核对`SYNC_MANIFEST.json`。包不含原图、标签、模型权重或旧结果，不覆盖这些资料。新模块不修改旧的三特征代理及扩散训练路径。

Python要求与项目一致（>=3.11）。CPU环境所需依赖见`requirements_mechanics_cpu.txt`，不依赖PyTorch、CUDA或diffusers；现有项目环境通常已具备这些CPU库。

需要的已有资料：

- `output/mechanics_m0/extraction/`中的CSV、extraction.json、window_features.npz；
- 当前manifest/compositions及其对应原图；
- 22合金力学主表及68试次表；
- 可选：服务器原manifest/compositions副本，用于解释此前字节哈希差异；
- 可选：人工核实的试次终止原因表，列为`alloy_id,ncorr_curve_file,endpoint_reason,evidence`。

终止原因只接受`confirmed_fracture`、`confirmed_correlation_loss`、`unknown`。确认项必须有证据文字；未提供一律unknown。终点=峰值只标记，不据此认定截断。终点应力比用现有“末段统计应力/峰值”，不能冒称最后一个原始数据点。

## 配置与预注册

复制`configs/mechanics_m0_v2_cpu.yaml`为`configs/mechanics_m0_v2_cpu.local.yaml`，按服务器路径修改，设置**全新**output_root。科学运行固定全21开发合金和至少1000次置换；`software_fixture`保持false。

默认预登记：

- 7个成分岭系数；图像惩罚为其1/3/10/30倍，另有明确的`"off"`候选，表示图像项禁用。选择忽略图像不保证整个嵌套模型的外层误差永远不劣于基线。
- 软特征保留原共享惩罚、PC1分块、原三维分块、仅图像四组，所有结果完整报告。
- 硬描述符主版本4项，最多另登记一个含布局量的变体，不能结果出来后无限增加。
- 相关剔除阈值0.9，按预先固定优先级保留，最多3个读出维度。不按不同单位的原始方差决定保留列。
- 原三条预测门不变：RMSE增量≥15%、至少ceil(12N/21)个合金绝对误差下降、删除任一合金完整重验后增量仍为正。
- 完整嵌套置换至少1000次，预登记经验尾概率阈值≤0.05；预测门和置换门同时满足才列为CPU候选。这个尾概率用于配对敏感性检查，不是严格的条件独立性证明。
- v2额外记录并默认要求MAE和Spearman不恶化；结果同时输出“原门”和“v2附加门”，明确这是第二轮新增约束，不回改v1判定。

准备并锁定：

```bash
python -m scripts.run_mechanics_m0_v2 --config configs/mechanics_m0_v2_cpu.local.yaml --phase prepare
```

只核验并写`plan.json`/`plan.sha256`，不进行回归或特征提取。锁定配置、原图、输入表、旧特征和代码哈希。实际使用的六维成分/标签必须与v1表一致；原图也必须匹配原提取时的哈希。服务器原表未提供时如实报告，不能假定哈希不同只是编码变化。

准备后改配置、数据或代码会被拒绝；要改需保留旧版本并新建预注册目录。失败阶段同样不覆盖，避免把不完整运行混入正式结果。

## 执行顺序

在项目根目录运行以下各阶段，或者prepare之后用`--phase all`串行执行这四个CPU阶段。`all`不自动prepare，也没有任何训练子命令。

```bash
python -m scripts.run_mechanics_m0_v2 --config configs/mechanics_m0_v2_cpu.local.yaml --phase audit
python -m scripts.run_mechanics_m0_v2 --config configs/mechanics_m0_v2_cpu.local.yaml --phase extract-hard
python -m scripts.run_mechanics_m0_v2 --config configs/mechanics_m0_v2_cpu.local.yaml --phase validate
python -m scripts.run_mechanics_m0_v2 --config configs/mechanics_m0_v2_cpu.local.yaml --phase decision
```

或：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.run_mechanics_m0_v2 --config configs/mechanics_m0_v2_cpu.local.yaml --phase all
```

1000次完整嵌套置换在多个表示上仍可能耗时较长；零GPU不代表没有CPU成本。程序逐目标、逐表示打印进度，不承诺完成时长。不要把置换次数降低后的软件试跑当科学放行结果。

### audit输出

- `soft_diagnostics.json`：各软表示折外RMSE/MAE/秩相关；有效秩、条件数、相关性及成分预测折内PC1的读数；ID21的截距/成分/图像贡献、成分/图像z值、范围外标记、岭杠杆与Cook-like量。
- PC1轴在各折重新拟合，成分到PC1的CV分数相对各折训练均值计算，不冒称同一个全局PC1上的普通R²或中介检验。
- 岭Cook-like只作固定变换/惩罚下的条件影响描述，不套用OLS显著性结论；外层留出残差按该折目标标准差归一化。
- `repeatability.json`与两个试次CSV：合金内方差、合金中位数间方差、近似ICC、合金中位数bootstrap波动、已确认断裂子集的重算中位数。
- 原表有68试次，主统计**只用21开发合金的65试次**，ID03的3条不参与阈值/噪声判断。原始68条不修改。
- 重复波动混合试样与测量差异，不是不可突破的预测RMSE下限。Bootstrap仅针对已观察到的少量试次，不能发现未知系统性DIC误差。
- 确认子集目前只提供敏感性读数，不自动换掉主标签重新跑一个有利结果；需要改标签时另立分析版本。
- `id21_originals_checklist.csv`：成分来源、载荷/DIC、EBSD扫描、热处理、试样视场对应等原件仍需回收，代码不会凭空填已确认。
- `table_comparison.json`：有服务器原表就逐开发合金/字段比对；没有原件就记录缺失和已有字节哈希。

### extract-hard输出

复用`native_rgb_visible_window_v1`的分区、边界膨胀、可见区域和小区域过滤口径，不resize，不改阈值选好结果。新指标schema为`native_rgb_hard_mechanics_v2`：

1. `log_area_weighted_diameter_px`：区域面积加权平均等效直径的自然对数。
2. `log_region_density_per_px2`：保留区域数/窗口总像素面积的自然对数。
3. `coarse_area_fraction`：保留区域中，直径≥预定20px的面积占比。
4. `area_weighted_aspect`：面积加权长宽比，惯性张量小像素修正沿用旧代码。

可选`centroid_density_cv`：默认64px网格中区域质心数量密度的变异系数。它只描述这一种布局代理，不能代表完整空间排列；网格面积不等时按实际面积计算。

每个窗口先计算变换后的描述符，再按合金取均值；直接复用v1固定窗口位置。无效窗单列，默认至少90%窗口有效才形成合金特征；达不到则保留失败与NaN，不偷偷删合金。输出逐窗质量、合金表和特征银行。

这些是RGB可见连通区域，不是CTF晶粒；px不等于μm。硬算法不可微，本轮不会把这些指标直接反传进U-Net。

### validate输出

每个目标/表示独立输出报告和所有外层模型。外层留一、内层调参，**标准化、PC1、相关剔除都只在对应内层训练折拟合**。仅成分基线同样严格调参；删除重验不用于删点放行；完整图像块置换重跑嵌套，不把经验尾概率解释成严格条件独立。

所有登记版本完整报告；不能只挑最优版本当独立确认。有效硬窗口不足时，硬路线报告阻断，不降低阈值或剔除合金。

### decision输出

`cpu_decision.json`使用与旧M0训练放行文件**不同的schema**，始终：

```json
{"training_authorized": false, "cnn_training_authorized": false}
```

下一步可能是：

- `consider_descriptor_distillation_after_review`：硬形貌预测门通过，供审阅后讨论CNN蒸馏；
- `review_regularized_soft_proxy_before_remaining_m0`：PC1/分块软代理通过，先审阅正则化解释；
- `need_data_or_segmentation_review`：标签/重复汇总或分区质量需要补查；
- `stop_registered_feedback_route_analysis_only`：本轮登记路线仍未通过；
- `software_fixture_no_scientific_decision`：仅合成软件测试，不能作为科学结果。

即使有候选通过，还没有自由生成域移、噪声区间、CNN梯度或物理验证，不能据此直接开C0/C1。CNN仅保留为下一阶段，不在本轮实现/训练。

## 软件验证与回传

`tests/test_mechanics_m0_v2_cpu.py`验证共享岭等价性、分块/关闭图像项、折内PCA/剔除、标签隔离、现有分区口径、ID21分解、试次隔离、预注册与完整合成CPU流程。没有用真实标签筛选新版本。

跑完后回传整个新output_root，尤其`plan.json`、`audit/`、`hard/`、`validation/summary.json`、逐目标验证报告与`cpu_decision.json`。保留原M0-v1结果，不覆盖。
