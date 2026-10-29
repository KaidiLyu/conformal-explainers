# Bamboo Steamer · Conformal Image Explainers

NYU CV 课程项目：复现 [Learning Conformal Explainers for Image Classifiers](https://arxiv.org/abs/2509.21209)，比较五种归因方法，并提供交互式结果查看器。

**当前版本是可运行的基础代码，不是已完成的学期项目，也不是作者官方实现。正式 Imagenette 训练与论文数值复现尚未完成。**

## 当前进度

| 模块 | 状态 |
|---|---|
| Imagenette 数据与固定划分 | 已下载并检查 13,394 张图片；共享清单与内容指纹已保存 |
| 四路数据划分、ResNet50 训练及保存 | 已实现；用小型测试验证，未进行完整 Imagenette 训练 |
| Saliency、Input × Gradient、GradientSHAP、KernelSHAP | 已接入 Captum，已在合成数据上运行 |
| FastSHAP | 已复用作者代码、修复兼容性；短训练和推理验证通过，未完成正式训练 |
| Pixelwise、Super-Pixels、Scaled Values 校准 | 已实现，包含关键数值测试 |
| 5 × 3 × 4 = 60 组实验流程 | 合成数据流程验证通过，不代表论文实验完成 |
| Streamlit Demo | 已实现预计算结果浏览；不是任意图片实时上传解释器 |
| 自动分析报告、曲线与失败清单 | 已实现；通过合成结果验证，等待正式实验结果 |
| GitHub 远程仓库 | [KaidiLyu/conformal-explainers](https://github.com/KaidiLyu/conformal-explainers)，公开仓库 |
| 正式结果、论文对照、失败分析、报告与展示 | 待完成 |

## 安装与快速运行

推荐 Python 3.12。在本仓库目录执行；路径有空格时请加引号。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,demo]'
pytest -q
conformal-cv smoke
streamlit run app.py
```

Demo 左侧填写 `outputs/smoke`，可以看到合成数据结果。`smoke` 不下载数据、不需要 GPU，会真正执行 FastSHAP 短训练和全部 60 组配置。为保护已有实验，重复执行需使用新目录，例如 `conformal-cv smoke --output outputs/smoke-2`。

Mac 可使用 CPU/MPS；CUDA 服务器请先按照 [PyTorch 官方说明](https://pytorch.org/get-started/locally/) 安装匹配驱动的 PyTorch，再安装本项目。KernelSHAP 和校准搜索计算量较大，正式实验建议先做小规模试跑。

## Imagenette 正式流程

### 1. 下载并固定划分

数据来自 [fastai/imagenette](https://github.com/fastai/imagenette)，下载也可使用 [torchvision Imagenette](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.Imagenette.html)：

```bash
python scripts/prepare_imagenette.py
```

标准版本按论文表 5 划分：训练 8,522、验证 947、校准 2,924、测试 1,001。作者没有公开具体样本清单，这里使用 seed=42 分层抽样，属于独立重建划分。组员使用仓库中的 `manifests/imagenette-seed42.json`，不要各自重新划分；准备脚本会核对压缩包、逐图可解码性和冻结指纹。数据与权重默认不纳入 Git。

Cloud Bursting 的真实图片试跑、Slurm 脚本和验收说明见 [HPC 试跑指南](docs/hpc_pilot.md)。

### 2. 训练分类器，再训练 FastSHAP

```bash
conformal-cv train --config configs/imagenette.yaml --device auto
conformal-cv train-fastshap --config configs/imagenette.yaml --device auto
```

分类器只根据验证集选择最佳 checkpoint。FastSHAP 使用冻结分类器的类别概率和零基线掩码训练，不接触校准集、测试集。模型文件保存数据划分指纹，FastSHAP 额外保存分类器文件指纹；不匹配时会拒绝继续评估。

配置中的 `epochs`、学习率和 FastSHAP 小型卷积架构只是基础设定，需用训练/验证数据确定，不是论文完整超参数。`train_limit`、`validation_limit` 可用于开发；正式运行应删除这些限制。

### 3. 校准与评估

```bash
conformal-cv evaluate --config configs/imagenette.yaml --device auto
streamlit run app.py
```

默认使用全部校准集和测试集。开发时复制配置，设置 `calibration_limit: 20`、`test_limit: 10`、较小 `threshold_steps`，并使用独立的 `output_dir`。这种小样本运行只能查错，不能验证 99% 置信水平。正式实验恢复完整配置。测试集不能用来调参数或选择最终配置。

输出包括：

- `summary.csv`：fidelity、Wilson 95% 区间、保留面积、分类准确率、归因耗时及校准总耗时。
- `per_image.csv`：原图预测、掩码后预测、真实标签和逐图指标。
- `thresholds.json`：每种解释器、校准方式、置信度对应的固定阈值。
- `*_calibration_scores.json`：用于审计的逐图校准分数，`null` 表示全图回退。
- `metadata.json`：配置、数据划分/模型指纹、平台、样本量和完成状态。
- `*.npz`：供 Demo 查看原图、归因图、保留区域和掩码后图片。

`calibration_seconds` 是该解释器的整段校准耗时，包含归因与三种校准搜索；同一方法的不同配置重复引用此总值，不能相加。归因计时在 GPU 上显式同步。FastSHAP 训练时间另存于 checkpoint。首次运行开销与正式严谨的重复计时仍需在实验中分析。

### 4. 生成分析报告与图表

```bash
conformal-cv report --results outputs/imagenette-v1
# 已有合成结果可直接试用：
conformal-cv report --results outputs/smoke
```

默认写入结果目录下的 `report/`；重复生成时使用新目录，例如加上 `--output outputs/smoke-report-v2`。原始实验文件不会被修改。命令只接受 `status: complete` 的运行，并核验所有配置的测试样本一致、没有重复/缺失记录，且预测与 agreement 字段相符。

- `report.md`：中文报告，包含各配置的指标、样本量与分析边界。
- `confidence_curves.png/.pdf`：目标置信度与实测 fidelity（95% Wilson 区间）、保留面积曲线。
- `attribution_timing.png/.pdf`、`timing.csv`：每种方法每张图只计一次的归因耗时，不包含训练与校准搜索成本。
- `metrics.csv`：从 `per_image.csv` 重新计算的指标，不依赖可能过期的 `summary.csv`。
- `failures.csv`：全部掩码前后预测改变记录，可按 `sample_id` 回查图片；同一图片可能出现多次。
- `provenance.json`：输入文件指纹与实验元数据。

合成结果的报告和图表会明确标注用途。不同方法曲线可能重叠，精确数值见指标表。报告不会自动选出“最佳方法”或用测试集调整参数。当前只提供失败案例索引，已有 NPZ 仍是评估时保存的前若干张示例。

## 核心代码导航

```text
configs/imagenette.yaml       共享实验配置
src/conformal_explainers/
  data.py                    数据划分、预处理、文件指纹
  models.py                  ResNet50、FastSHAP 基础网络
  explainers.py              五种方法的统一接口
  calibration.py             三种区域选择与阈值校准
  training.py                两类模型的训练与加载
  experiment.py              独立校准/测试、指标与结果保存
  reporting.py               结果校验、统计、图表与中文分析报告
  smoke.py                   不依赖真实数据的集成检查
  cli.py                     命令行入口
src/fastshap/                带 MIT 许可证的上游代码子集
tests/                      数值、归因、训练和数据划分测试
app.py                      预计算结果查看器
docs/                       复现决策、团队任务、验证记录
```

项目自编核心代码有中文注释；上游代码保留原始英文注释与版权。详见 [复现决策](docs/reproduction_notes.md)、[团队任务](docs/team_plan.md)、[第三方来源](THIRD_PARTY_NOTICES.md)。

## 后续必须完成的工作

1. 确认课程要求、最终分工、GPU 和时间预算。
2. 在正式数据上训练，冻结模型与解释器配置；检查准确率和 FastSHAP 验证损失。
3. 固定 rho 规则、网格/超像素参数及搜索预算，运行完整评估。
4. 与论文表 1/2 的 Imagenette 行对照；制作置信度曲线、失败案例，讨论未达到目标的结果。
5. 根据时间预算进行重复校准划分/不确定性分析，完善 Demo 和展示。
6. 编写报告、引用和 AI 辅助开发说明；每位组员理解自己负责的模块。

不能把“代码运行成功”写成“已复现论文结论”。本实现对模型、归因粒度和校准细节的选择都需要纳入最终报告。

