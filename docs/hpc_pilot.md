# Cloud Bursting 首次真实图片试跑

2026-10-29：已登录 OOD；已提交 1 小时、单 L4 的 Jupyter 会话，作业号 14720。用户预计晚上 9 点多可分配资源；尚无 GPU 实测结果。以下步骤用于该会话或之后的批作业，不要同时重复申请两份 GPU。

本地试跑快照包：`outputs/hpc/cv-cloud-pilot-20261029.zip`。含代码、清单与说明，不含图片、权重或本机虚拟环境。浏览器上传尚未完成；云端部署优先从 GitHub 拉取本次更新，以避免使用较旧的本地快照包。在 `/scratch/<NetID>` 中执行：

```bash
git clone https://github.com/KaidiLyu/conformal-explainers.git
cd conformal-explainers
# 若已克隆，在该仓库内执行 git pull --ff-only。
```

确认 `scripts/gpu_pilot.py` 和 `manifests/imagenette-seed42.json` 存在后，再按下文安装环境和下载数据。

## 已完成的数据准备

官方 Imagenette 320px 压缩包已在本地下载并通过 torchvision 提供的 MD5 验证。13,394 张图片全部可解码；路径无交叉；未发现文件内容完全相同的图片。内容指纹仅检测精确重复，不代表已排除所有近似重复。

固定清单：`manifests/imagenette-seed42.json`。它仅保存图片相对路径和标签，适合组员共享；图像与模型不提交 Git。配置已指向该清单。

| 划分 | 数量 |
|---|---:|
| train | 8,522 |
| validation | 947 |
| calibration | 2,924 |
| test | 1,001 |

- 清单 SHA256：`4191e096d87597f146f5fc6c5da2b9ec2eb071f075db5c8479ac3742f67a22e3`
- 数据内容 SHA256：`0c86ec5376de254e8b09e1faec7e05cdb01cf3b7d115a4cb20853d7ef0eba87b`
- 压缩包 SHA256：`569b4497c98db6dd29f335d1f109cf315fe127053cedf69010d047f0188e158c`

完整审计在 `manifests/imagenette-seed42.audit.json`。`write_splits` 会拒绝覆盖内容不同的已有清单。HPC 应直接使用同一清单，不要重新随机划分。

## 运行环境与数据

课程要求详见 [Cloud Bursting 说明](https://brightspace.nyu.edu/d2l/le/lessons/627988/topics/14038330)。本次已实际读取该文档，确认：

- 账户 `csci_ga_2271-2026fa`。
- 单 L4 分区 `g2-standard-12`，GPU 参数 `--gres=gpu:1`。
- 单 A100 分区 `c12m85-a100-1`；本次先用 L4。
- CPU 分区 `n2c48m24` 可用于环境/数据准备。
- 持久文件保存于 `/scratch/<NetID>`；批作业加 `--requeue`。
- 低 GPU 利用率持续 20 分钟可能关闭节点；课程计算只使用 Cloud Bursting。

环境尚未在远端安装或验证。先在课程支持的 Python/Conda/Singularity 环境中准备 PyTorch CUDA 与本项目依赖，避免 GPU 会话长时间只下载/安装软件。不要直接复制本机 macOS `.venv` 到 Linux。

在云端项目根目录执行：

```bash
python -m pip install -e '.[dev]'
python scripts/prepare_imagenette.py
# 预先缓存正式权重，避免试跑时把下载算作模型运行耗时。
TORCH_HOME="$PWD/data/torch-cache" python -c 'from torchvision.models import resnet50, ResNet50_Weights; resnet50(weights=ResNet50_Weights.DEFAULT)'
```

准备脚本会下载数据、检查每个文件，并与冻结审计对照；任何差异都会报错，不能随意覆盖审计来跳过检查。冻结清单/指纹一致后才运行试跑。

## 在已经获得的 GPU 会话运行

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
python -u scripts/gpu_pilot.py --output outputs/gpu-pilot-l4-01
```

默认使用 224×224 真实图片、ImageNet 预训练 ResNet50 加随机 10 类分类头、FP32；训练 5 步，每批 8 张。验证只看开发用的前 10 张图，归因/搜索测其中 3 张；不访问最终校准和测试图片，也不声称这几步已训练好模型。

当前归一化仍为 ImageNet 统计。与论文训练集统计的对齐属于后续正式训练任务，本次记录这一差异。

若没有交互会话，可从项目根目录提交 **另一种运行方式**：

```bash
mkdir -p outputs
sbatch scripts/hpc/pilot.slurm
```

脚本申请单 L4、30 分钟，沿用提交时的已安装 Python 环境；可用 `CV_PYTHON` 指定解释器路径。Slurm 参数已按课程文档编写且通过本地 shell 语法检查，尚未实际提交这个批脚本。

## 保存与验收

- `pilot.json`：设备、版本、清单指纹、每步训练 loss、耗时、显存峰值、归因与搜索记录。
- `summary.md`：自动汇总的中文性能表。
- `pilot_checkpoint.pt`：每个训练步骤保存权重、优化器和完成步数；仅供试跑检查。

JSON 和 checkpoint 使用临时文件后替换，避免写入中断破坏上一份文件；已有输出目录不会被覆盖。**当前短试跑不提供精确断点续跑**，重排后从头试跑，使用新的 attempt 目录保留旧结果。`--requeue` 本身不等于程序恢复；正式训练的完整恢复仍须另做。

训练计时排除首步预热、图片读取/传输及写盘；归因每种方法预热一次。搜索分别记录 Pixelwise、Super-Pixels、Scaled Values。显存来自 PyTorch 分配器，不等于整张卡的 nvidia-smi 占用。

本阶段通过标准：CUDA 可用、训练 loss/梯度和归因值有限、完整输出 `status=complete`，具备训练与解释耗时、显存峰值，并能核对输入清单指纹。实测结果出来前不填造 GPU 性能数字。
