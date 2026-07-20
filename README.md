# QueryPro：稀疏 KV Cache 自投机解码实验

本仓库研究长输入、短输出场景下的稀疏 KV Cache 自投机解码。Draft
阶段由同一个大语言模型仅访问部分 KV Cache生成候选 token，Verification
阶段再使用完整 KV Cache验证候选。该方案只需要维护一份 KV Cache，适合对
显存和内存更敏感的端侧推理。

当前研究问题已经从“预测完整的未来 Query”逐渐收敛为：

> 在固定保留10% KV页面的条件下，能否用极低成本判断每个 draft位置应加入
> 哪1～3个关键页面，从而缓解一轮8个 token内部的 selection aging？

## 评价原则

最终目标是让更多 draft token通过完整模型验证，因此主要指标始终是：

- 平均接受长度（Mean Accepted Length）；
- 8-token draft完整接受率；
- 第1～8个 draft位置的条件接受率。

KV集合重合度、candidate recall和 attention recovery只用于解释结果，不能代替
真实的 `sparse draft + dense verification` 接受指标。已有实验已经表明，接近
最优的 attention coverage并不必然带来接近最优的 logits或 top-1输出。

## 当前主实验设置

- 模型：本地 Qwen3-4B；
- 数据：LongBench五个长上下文短答案 QA数据集；
- 输入长度：正式实验统一截取到8192 tokens；
- 最大生成长度：64 tokens；
- Draft长度：8；
- 解码：greedy，batch size 1；
- KV预算：保留10%的历史页面；
- Page size：16 tokens；
- 选择粒度：每层、每个 Query head独立选择页面，并正确映射 GQA KV head；
- 页面分数：页内最大的 post-RoPE QK分数；
- 无额外 sink、recent window或预算外固定页面。

早期 Query预测、union和 frontier实验使用的是 ReasoningData、较短上下文及代表
层/头，只用于探索问题结构，不能与后续 LongBench端到端接受长度直接混合比较。

## 实验脉络

### 1. 预测未来 Query

最初尝试使用参数极少的 Temporal Linear和 Tiny TCN，根据历史 Query预测约8个
token之后的 Query，再据此选择 KV。Held-out预实验中，学习模型没有超过简单的
`pre-RoPE corrected endpoint mean`：后者的 changed-step attention recovery为
`0.8459`，Temporal Linear为`0.8349`，Tiny TCN为`0.8281`。这说明直接预测完整
Query并不是当前最有希望的优化目标。

### 2. 分析未来 KV需求与局部 frontier

未来8个真实 Query的 oracle KV union约为单 Query预算的`2.68×`，说明不同位置
需要的 KV集合确实不同。但上一轮 selection遗漏的重要 KV并没有集中在一个很小
的旧排名 frontier中：5% frontier只恢复约30.6%的遗漏 attention mass。受限候选
页重排的最佳 joint near-full rate也只有`0.6823`，因此“直接重排静态 cutoff附近
的一小段页面”不足以解决问题。

### 3. 验证动态选择是否提高真实接受长度

在统一的 page-level 10%预算下，比较轮内固定的 endpoint mean与逐 draft-token
Oracle B：

| 方法 | 平均接受长度 | 完整8-token接受率 |
| --- | ---: | ---: |
| Static endpoint mean | 4.7655 | 27.71% |
| Oracle B | 6.5419 | 66.74% |

Oracle B在每个 draft位置使用当前真实 Query重新选择页面。结果证明，静态集合随
draft推进而过时，是限制接受长度的重要因素。

### 4. 排除“只是静态方法没有选好”的解释

Best Static Oracle事后读取未来8个真实 Query，寻找一套总体 attention coverage
最大的固定10%页面。其平均接受长度为`5.4385`，仍明显低于 Oracle B的`6.5419`。

更关键的是，Best Static已经达到逐 Query attention oracle约`98.96%`的总体
coverage，却只恢复了约35.8%的 Oracle B接受长度收益。这说明：

- 一套公共页面难以同时服务整轮 Query；
- attention mass不是充分的 acceptance目标；
- 后续方案需要逐位置刷新，并关注少量对 hidden state和 logits敏感的页面。

### 5. 验证少量增量更新的上限

Token-level实验首先给出了积极信号，但它与既有 page-level基线存在粒度差异，
因此只作为探索性结果。随后补充了严格统一口径的 page-level Oracle Incremental
实验：每轮从 endpoint mean集合开始，在第2～8个 draft位置最多替换所选页面
预算的1%、5%、10%或20%。三组历史基线直接复用，没有重复推理。

| 方法 | 平均接受长度 | 完整8-token接受率 | 每步实际替换页数 | Oracle B收益恢复（sample macro） |
| --- | ---: | ---: | ---: | ---: |
| Static endpoint mean | 4.7655 | 27.71% | 0 | 0% |
| Best Static Oracle | 5.4385 | 37.10% | 0 | 35.80% |
| Page Incremental 1% | 6.2854 | 58.43% | 0.99 | 89.15% |
| Page Incremental 5% | 6.3348 | 62.22% | 2.89 | 92.66% |
| Page Incremental 10% | 6.3515 | 62.81% | 5.29 | 93.40% |
| Page Incremental 20% | 6.4404 | 64.91% | 7.85 | 96.34% |
| Oracle B | 6.5419 | 66.74% | 全量重选 | 100% |

典型8192-token输入有512页，10%预算为52页。因此1%更新对应每层、每个 Query
head、每步最多替换1页，却已经恢复约89%的 Oracle B收益；5%约替换3页，是当前
更稳健的质量—开销折中点。这为研究轻量的 page entrant predictor提供了直接动机。

## 目录说明

| 目录 | 内容 | 主要结论或作用 |
| --- | --- | --- |
| [`query_forecast/`](query_forecast/) | Query采集、数据划分和 Temporal Linear/Tiny TCN预测器模块；根目录的 [`run_query_forecast_experiment.py`](run_query_forecast_experiment.py) 是入口 | 学习式完整 Query预测没有超过 endpoint mean基线 |
| [`docs/`](docs/) | 早期 Query Forecast实验简报 | 保存早期实验设置、成本和 held-out结果 |
| [`future_query_union_frontier/`](future_query_union_frontier/RUN.md) | 统计未来8个 Query的 oracle KV union及旧排名遗漏的 frontier集中程度 | Union约为2.68倍预算，小 frontier无法覆盖多数遗漏重要性 |
| [`restricted_frontier_rerank/`](restricted_frontier_rerank/RUN.md) | 用真实 Query只重排旧 ranking附近的少量候选页 | 候选覆盖不足，未稳定接近全量 query-aware selection |
| [`oracle_b_vs_static/`](oracle_b_vs_static/RUN.md) | 完整自投机解码中比较 Static endpoint mean与逐位置 Oracle B | 动态页面选择将平均接受长度从4.77提高到6.54 |
| [`best_static_oracle/`](best_static_oracle/RUN.md) | 比较 Static、最优固定公共集合与 Oracle B | 即使最优静态 coverage接近99%，接受长度仍明显落后 |
| [`token_incremental_selection/`](token_incremental_selection/RUN.md) | Token级少量增量替换上限和候选池探索 | 提供方向性证据；因选择粒度不同，不与 page-level主结果直接混用 |
| [`page_incremental_selection/`](page_incremental_selection/RUN.md) | 与三个主基线完全同口径的1%/5%/10%/20%页面增量更新 | 每步更新1页已恢复约89%的 Oracle B收益，是当前主实验结论 |
| `scipy/`、`sklearn/` | 为规避服务器上不兼容的可选依赖而提供的最小 import stub | 不是完整 SciPy/scikit-learn，也不应作为通用实现使用 |

每个独立实验目录均包含 `RUN.md`，说明实验定义、正确性检查、服务器命令、输出
字段和结论判断。正式结果保存在对应目录的 `results/` 下。

## 当前结论与下一阶段

预实验支持以下主线：

1. 10% KV预算本身不是主要瓶颈，关键是选择更适合当前 Query的10%；
2. 不存在一套足够好的固定10%公共页面可以完全替代逐位置选择；
3. 无需每步全量更换，1～3个关键 entrant页面已经能恢复大部分接受长度收益；
4. 下一阶段应研究轻量 page entrant predictor，而不是继续预测完整未来 Query。

下一阶段的预测器只能使用在线可获得的因果信息，例如上一轮 verification重要性、
当前 Query与小候选池 Key的相似度、Query drift、页面位置和当前 selection边界特征；
不能使用下一位置真实 Query、attention、hidden state或 logits。模型选择和最终比较
仍以完整自投机解码的平均接受长度为准。

## 运行与复现

默认本地资源位置为：

```text
D:\preExperiments\model\Qwen3-4B
D:\preExperiments\ReasoningData
D:\preExperiments\LongBench\data
```

仓库不会自动下载或替换模型与数据。运行任一正式实验前，应先执行该目录
`RUN.md`中的小样本 correctness test，并确认：

- patched dense path与原模型 top-1一致，logit误差在阈值内；
- 不同方法最终提交相同的 dense-greedy输出序列；
- KV预算、page mask、position id、RoPE和 GQA映射正确；
- 每个样本、每种方法均从独立的干净 dense cache开始。

Qwen/Transformers内部 attention与 Cache接口可能随版本变化。服务器正式运行时应
固定已验证的依赖版本，不能仅凭 forward成功就认为稀疏 attention实现正确。
