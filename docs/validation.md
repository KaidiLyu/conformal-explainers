# 基础版本验证记录

验证日期：2026-10-28。平台：macOS ARM64，Python 3.12，CPU。

- `pytest -q`：22 项通过。覆盖阈值方向、并列值、零/负归因、常数图、rho 两种含义、非单调预测、分层划分、四种 Captum 后端、FastSHAP 真训练与效率约束、ResNet50 反向传播。
- `conformal-cv smoke`：五种方法 × 三种校准方式 × 四个置信度，共 60 条汇总、240 条逐图记录。FastSHAP 在合成数据上训练 2 个 epoch。
- `ruff check` 与 `ruff format --check`：通过；第三方源码不纳入项目风格检查。
- Streamlit AppTest：默认空状态、合成实验结果读取、两张指标表均通过。
- GitHub Actions：已提供 CPU 工作流，尚未在远程执行。

合成验证使用 12 张训练、4 张验证、8 张校准、4 张测试图片，以及只依赖局部红色通道的简单分类器。它不评估真实视觉任务，也不能检验 99% 可靠性。真实 ResNet50 的训练能力另外通过一次前向、交叉熵和反向传播检查。

未完成：完整 Imagenette 下载/训练、FastSHAP 正式训练、60 组正式数据实验、论文数值复现、GPU 性能测量与最终报告。不能用合成数据指标替代这些工作。

当前精确依赖快照见 `requirements-tested-macos.txt`；该文件用于审计本机环境，CUDA/Linux 环境应采用对应平台 PyTorch 并重新记录依赖，而不是机械复制本机锁定版本。

## 2026-10-29：结果分析模块

- 新增 `conformal-cv report`，从已有完整实验生成中文 Markdown 报告、PNG/PDF 图表、指标表、失败清单和输入指纹。
- `pytest -q`：33 项通过（新增 11 项）。覆盖 Wilson 区间计算结果、跨配置计时去重、重复/缺失样本、标签/预测不一致、非法数值、未完成运行、空失败清单及防止覆盖已有报告。
- `ruff check` 与 `ruff format --check`：通过。
- 实际执行 `conformal-cv report --results outputs/smoke`：成功读取已有 60 组配置、240 条逐图记录，导出 60 条指标与 49 条失败记录；两张 PNG 图已目视检查布局。
- GitHub Actions 已加入 smoke 后生成报告的步骤；本次未验证远程 CI。

本次复用已有合成实验结果，没有重新训练模型或运行真实 Imagenette 实验。新增的报告工具不代表正式实验分析已完成。

## 2026-10-29：GPU 前置准备

- 官方 Imagenette 320px 数据已下载；13,394 张图片逐张解码通过；固定四路划分无路径交叉、无内容完全重复图片，冻结清单及审计存于 `manifests/`。
- 单元测试增至 36 项通过，增加划分防覆盖、预热排除/计时分组、非有限值写入保护和无 CUDA 时拒绝冒充 GPU 测试的检查。
- 新增 `scripts/gpu_pilot.py` 和课程 Cloud Bursting L4 批作业模板，记录显存与各阶段耗时、自动中文汇总并逐步保存试跑权重。批脚本通过 `bash -n`，尚未运行远端 GPU 测试。
- OOD 已登录；已提交 1 小时 L4 Jupyter 会话（14720），截至本次准备阶段仍等待 GPU。
- 真实图片本地 CPU 检查：2 张 224×224 训练图片完成一次 ResNet50 前向/反向与参数更新，1 张验证图片生成有限的 Saliency 归因；记录在 `outputs/local-real-image-check.json`。使用随机初始化，仅检查链路，不用于模型效果或 GPU 性能结论。
