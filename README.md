# CARE-PD Prediction Module

本项目用于基于人体步态骨架序列预测帕金森 UPDRS 评分类别。当前推理流程以 HumanML3D 263 维 `.npy` 动作特征为输入，使用 MoMask backbone 和训练好的分类头输出预测类别、各类别概率和 logits。同时项目提供了一个 SMPL `.pkl` 到 HumanML3D `.npy` 的转换脚本，以及一个 Streamlit 前端用于上传、预览和分析步态序列。

## 功能概览

- 将 SMPL 参数文件转换为每段步行对应的 HumanML3D 263 维 `.npy` 文件。
- 对 HumanML3D `.npy` 骨架序列进行 UPDRS 类别预测。
- 自动按配置裁剪或补齐到模型需要的序列长度，默认 `source_seq_len=196`。
- 使用训练数据统计量进行归一化。
- Streamlit 页面支持骨架预览、步态特征图表、类别概率可视化和 JSON 结果查看。

## 项目结构

```text
.
|-- assets/
|   |-- Pretrained_checkpoints/momask/   # MoMask backbone 与分类模型权重
|   `-- stats/HumanML3D_norm_data/       # HumanML3D 归一化统计量
|-- configs/momask/
|   `-- inference_config.json            # 推理配置
|-- data/preprocessing/                  # SMPL 与 HumanML3D 预处理相关代码
|-- model/                               # 模型结构与 backbone 加载代码
|-- outputs/                             # 示例输出目录
|-- utility/                             # 动作恢复与辅助工具
|-- smpl_pkl_to_humanml3d_npy.py         # SMPL pkl 转 HumanML3D npy
|-- inference.py                         # 命令行推理入口
|-- care_pd_frontend.py                  # Streamlit 前端
`-- requirements.txt                     # Python 依赖
```

## 环境准备

建议使用 Python 3.10，并在独立虚拟环境中安装依赖。

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如果需要使用 GPU，请根据本机 CUDA 版本单独安装匹配的 PyTorch 版本。当前 `requirements.txt` 中的 PyTorch 是 CPU 版本。

## 必要文件

运行推理前请确认以下文件存在：

- `configs/momask/inference_config.json`
- `assets/Pretrained_checkpoints/momask/best_model.pt`
- `assets/Pretrained_checkpoints/momask/net_best_fid.tar`
- `assets/Pretrained_checkpoints/momask/opt.txt`
- `assets/stats/HumanML3D_norm_data/BMCLab/Mean.npy`
- `assets/stats/HumanML3D_norm_data/BMCLab/Std_FEAT_BIAS_included.npy`

说明：

- `best_model.pt` 应包含 CARE-PD 分类头权重，用于最终 UPDRS 预测。
- `net_best_fid.tar` 是 MoMask 预训练 backbone 权重，由配置文件中的 `model_checkpoint_path` 使用。
- 不要把 `net_best_fid.tar` 当作 `--model_path` 传给推理脚本，否则会缺少分类头权重。

## 输入数据格式

推理脚本接受 HumanML3D 格式 `.npy` 文件：

- 二维数组：`(n_frames, 263)`
- 三维数组：`(n_augmentations, n_frames, 263)`，脚本会默认取第一个增强版本

脚本会根据配置中的 `source_seq_len` 自动处理长度：

- 序列长于目标长度时，默认截取中间片段。
- 序列短于目标长度时，在末尾补零，并生成有效帧 mask。

## 从 SMPL pkl 转换为 HumanML3D npy

如果已有 SMPL 参数 `.pkl`，可以先转换为推理需要的 `.npy`：

```bash
python smpl_pkl_to_humanml3d_npy.py ^
  --input carepd_promptHMR.pkl ^
  --output outputs/humanml3d_npy ^
  --dataset BMCLab
```

可选参数：

- `--fps`：原始视频或 SMPL 序列帧率；不设置时会按数据集名称推断。
- `--target_fps`：目标帧率，默认 `30`。
- `--slope_correction`：强制执行坡度校正。
- `--model_path`：SMPL neutral body model 路径，默认 `data/preprocessing/common/body_models/smpl/SMPL_NEUTRAL.pkl`。
- `--template_path`：HumanML3D 目标骨架模板，默认 `data/preprocessing/transforms/000021.npy`。

支持的 `.pkl` 结构包括：

```python
{
    subject_id: {
        walk_id: {
            "pose": (T, 72),
            "beta": (10,) 或 (T, 10),
            "trans": (T, 3)
        }
    }
}
```

或单条序列：

```python
{
    "pose": (T, 72),
    "beta" 或 "betas": (10,) 或 (T, 10),
    "trans": (T, 3)
}
```

## 命令行推理

对单个 HumanML3D `.npy` 文件运行预测：

```bash
python inference.py ^
  --config configs/momask/inference_config.json ^
  --model_path assets/Pretrained_checkpoints/momask/best_model.pt ^
  --input outputs/humanml3d_npy/prompthmr_1__video_track1.npy ^
  --output outputs/prediction.json
```

输出 JSON 示例：

```json
{
  "predicted_updrs_score": 1,
  "class_probabilities": [0.12, 0.74, 0.14],
  "logits": [0.1, 1.9, 0.25],
  "input_file": "outputs/humanml3d_npy/prompthmr_1__video_track1.npy",
  "model_path": "assets/Pretrained_checkpoints/momask/best_model.pt",
  "config_path": "configs/momask/inference_config.json",
  "device": "cpu",
  "dataset": "BMCLab",
  "backbone": "momask"
}
```

其中 `predicted_updrs_score` 是模型预测的类别编号，类别数量由配置文件和模型分类头决定。

## 启动前端页面

项目提供 Streamlit 前端，可以上传 `.npy` 文件并查看骨架预览、步态图表和预测结果：

```bash
streamlit run care_pd_frontend.py
```

打开终端输出的本地地址后，在页面中：

1. 检查侧边栏中的配置路径和模型路径。
2. 上传 HumanML3D `.npy` 文件。
3. 查看骨架预览和运动特征。
4. 点击 `Predict UPDRS` 运行预测。

## 常见问题

### 提示 checkpoint 不包含分类头

请确认 `--model_path` 使用的是训练后的分类模型，例如：

```text
assets/Pretrained_checkpoints/momask/best_model.pt
```

不要使用 MoMask 预训练 backbone 文件：

```text
assets/Pretrained_checkpoints/momask/net_best_fid.tar
```

### 提示归一化文件不存在

检查配置文件中的：

```json
"humanML3D_normalization_data_path": "assets/stats/HumanML3D_norm_data",
"dataset": "BMCLab"
```

对应目录下应存在：

```text
assets/stats/HumanML3D_norm_data/BMCLab/Mean.npy
assets/stats/HumanML3D_norm_data/BMCLab/Std_FEAT_BIAS_included.npy
```

### 输入维度不正确

推理脚本只接受 HumanML3D 263 维特征文件，即 shape 为 `(n_frames, 263)` 或 `(n_augmentations, n_frames, 263)` 的 `.npy`。如果输入是 SMPL `.pkl`，请先运行转换脚本。

## 备注

该模块输出的是模型预测结果，不能替代临床诊断。实际使用时应结合医生评估、采集协议和模型验证结果进行解释。
