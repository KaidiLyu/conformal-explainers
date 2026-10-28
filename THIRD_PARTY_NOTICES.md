# 第三方代码与引用

## FastSHAP：实际复制并使用

- 来源：https://github.com/iancovert/fastshap
- 固定提交：`83a02d1fc1c214c0a0cb29116339e48b122dbeba`
- 作者：Ian Covert 等；版权文本 Copyright (c) 2021 Ian Covert。
- 许可证：MIT，原文保存在 `src/fastshap/LICENSE`。
- 复制文件：`fastshap.py`、`utils.py`、`image_imputers.py`；本地 `__init__.py` 仅导出 FastSHAP。
- 未复制上游 notebooks、数据集或训练权重。

本地修改：

1. 删除新版 PyTorch 不再支持的 ReduceLROnPlateau `verbose` 参数。
2. 为 ShapleySampler 的 NumPy Generator 分别设置验证/训练种子；仅设置 `torch.manual_seed` 不能控制这个 Generator。
3. `shap_values` 计算全特征预测时将 tensor 包装为 TensorDataset，修复批量 DataLoader 解包错误。

本项目不是原论文官方实现。上游代码的版权和许可不受本地新增代码影响。

## 通过依赖调用，未复制源码

- [Captum](https://github.com/pytorch/captum)：Saliency、InputXGradient、GradientShap、KernelShap。
- [PyTorch / torchvision](https://github.com/pytorch/vision)：分类器、训练、预处理。
- [scikit-image](https://github.com/scikit-image/scikit-image)：SLIC 分割。
- [scikit-learn](https://github.com/scikit-learn/scikit-learn)：分层划分、Captum 的回归依赖。
- [Streamlit](https://github.com/streamlit/streamlit)：本地结果查看器。

依赖包各自的许可证随其发行包分发。数据集和预训练权重不打包进入本仓库。

## 学术引用

```bibtex
@misc{alkhatib2025conformal,
  title={Learning Conformal Explainers for Image Classifiers},
  author={Amr Alkhatib and Stephanie Lowry},
  year={2025},
  eprint={2509.21209},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
@inproceedings{jethani2022fastshap,
  title={FastSHAP: Real-Time Shapley Value Estimation},
  author={Neil Jethani and Mukund Sudarshan and Ian Covert and Su-In Lee and Rajesh Ranganath},
  booktitle={ICLR},
  year={2022}
}
```

## 本地新增代码

新增代码由 Bamboo Steamer 课程项目使用，基础版本通过 AI 辅助开发。最终提交应按课程要求披露，组员需要审阅、理解、验证自己的模块。暂不替全体组员决定新增代码的公开开源许可证；远程默认拟设私有。

