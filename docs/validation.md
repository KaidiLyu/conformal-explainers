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
