# 第08步：完整代表性LOO留一验证

## 这一步验证什么

LOO会把一种合金的图像和成分从训练集中同时移除，再用其余21种合金训练模型。训练完成后，程序输入被移除合金的成分，生成8张图并与真实EBSD图进行比较。

PyCharm预设验证4种代表性合金：

    3、14、18、21

## 完整流程

每一折依次执行：

1. 排除当前合金并重新训练VAE；
2. 排除当前合金并重新训练条件扩散模型；
3. 输入被排除合金的成分并生成8张图；
4. 将8张生成图与该合金的真实图进行统计评价；
5. 保存单折报告。

四折全部完成后，程序生成JSON和CSV汇总报告。

## 在哪里看结果

    08_训练与生成结果/05_留一验证/
    |- loo_progress.json             当前折、阶段、用时和剩余时间
    |- loo_state.json                中断恢复状态
    |- fold_03/                      3号合金结果及严格留一模型
    |- fold_14/                      14号合金结果及严格留一模型
    |- fold_18/                      18号合金结果及严格留一模型
    |- fold_21/                      21号合金结果及严格留一模型
    |- loo_summary.json              四折完整汇总
    +- loo_summary.csv               便于Excel查看的指标表

每个 fold_XX 文件夹包含 models_strict、生成图、评价JSON和 fold_summary.json。

总流程当前运行到哪一折、哪一阶段，看 loo_progress.json 和 loo_state.json。当前折VAE或扩散模型的具体步数、loss、显存、已用时间和剩余时间，分别看：

    fold_XX/models_strict/vae/run_progress.json
    fold_XX/models_strict/vae/training_metrics.csv
    fold_XX/models_strict/diffusion/run_progress.json
    fold_XX/models_strict/diffusion/training_metrics.csv

同样的信息也会实时显示在PyCharm的运行窗口中。

## 中断后怎么办

重新运行同一个PyCharm配置 08_代表性LOO验证。程序会：

- 跳过已经完成且参数一致的折；
- 跳过已经完成的VAE或扩散模型；
- 从当前训练阶段的最新检查点继续；
- 跳过已经完成且参数一致的生成和评价。

不要手动删除检查点，也不要修改 fold_summary.json。

## 时间说明

当前电脑上，单个基础VAE约1.5小时，单个扩散模型约1.3小时。严格四折LOO需要重新训练4个VAE和4个扩散模型，预计约11至13小时。实际时间以PyCharm和 loo_progress.json 为准。

## 科学边界

- PyCharm配置启用了 --retrain-vae，因此是真正的图像与成分双重留出；
- 留出成分若超出其余21种合金的训练邻域，生成图文件和报告会明确标记为外推；
- 汇总报告只提供诊断指标，不自动给出“通过/不通过”，最终阈值需要在研究方案中预先确定；
- LOO完成并审核通过前，不运行第09步704张批量生成。
