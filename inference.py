import os
import argparse
import torch
import numpy as np
import json
import re
from pathlib import Path
import sys

from model.backbone_loader import load_pretrained_backbone
from model.motion_encoder import MotionEncoder
from const import path, const

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_CHECKPOINT_MODEL_KEYS = ('model', 'state_dict')


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def _extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in _CHECKPOINT_MODEL_KEYS:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    return checkpoint


def _infer_num_classes(config, state_dict):
    if isinstance(state_dict, dict):
        fc_weight_layers = []
        for key, value in state_dict.items():
            match = re.match(r'^head\.fc_layers\.(\d+)\.weight$', key)
            if match and hasattr(value, 'shape'):
                fc_weight_layers.append((int(match.group(1)), value))
        if fc_weight_layers:
            _, last_weight = max(fc_weight_layers, key=lambda item: item[0])
            return int(last_weight.shape[0])
    return int(config.get('num_classes', 5))


def _has_classifier_head(state_dict):
    return isinstance(state_dict, dict) and any(key.startswith('head.') for key in state_dict)


def load_model(config, model_path):
    """加载模型"""
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请下载预训练模型，执行以下命令:")
        print("bash scripts/download_models.sh")
        exit(1)
    
    # 加载骨干网络
    try:
        backbone = load_pretrained_backbone(config, 'momask')
    except Exception as e:
        print(f"❌ 加载骨干网络失败: {e}")
        print("请确保预训练模型已正确下载")
        exit(1)
    
    checkpoint = torch.load(model_path, map_location=_DEVICE)
    state_dict = _extract_state_dict(checkpoint)

    if not _has_classifier_head(state_dict):
        raise ValueError(
            "Checkpoint does not contain CARE-PD classifier head weights. "
            "Please use a trained classifier checkpoint such as "
            "assets/Pretrained_checkpoints/momask/best_model.pt or an "
            "experiment_outs/.../latest_epoch.pth.tr file, not the MoMask "
            "pretraining checkpoint net_best_fid.tar."
        )

    # 使用checkpoint中的分类头维度，避免配置和权重类别数不一致
    num_classes = _infer_num_classes(config, state_dict)
    print(f"使用配置文件中的num_classes: {num_classes}")
    
    # 创建完整模型，使用配置的num_classes
    model = MotionEncoder(
        backbone=backbone,
        params=config,
        num_classes=num_classes,
        train_mode='classifier_only'
    )
    
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected_head_keys = [key for key in incompatible.unexpected_keys if key.startswith('head.')]
    missing_head_keys = [key for key in incompatible.missing_keys if key.startswith('head.')]
    if missing_head_keys or unexpected_head_keys:
        raise RuntimeError(
            "Classifier checkpoint is incompatible with the current config. "
            f"Missing head keys: {missing_head_keys}; unexpected head keys: {unexpected_head_keys}"
        )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "⚠️ 非分类头权重存在不完全匹配，通常是backbone预训练键名差异: "
            f"missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}"
        )
    print("✅ 模型权重加载成功")
    
    model.to(_DEVICE)
    model.eval()
    
    return model

def _clip_or_pad_sequence(pose_data, target_len, select_middle=True):
    curr_len = pose_data.shape[1]
    valid_len = min(curr_len, target_len)

    if curr_len > target_len:
        if select_middle:
            middle_frame = curr_len // 2
            start_frame = middle_frame - (target_len // 2)
            pose_data = pose_data[:, start_frame:start_frame + target_len, :]
        else:
            pose_data = pose_data[:, :target_len, :]
        valid_mask = np.ones(target_len, dtype=np.float32)
    elif curr_len < target_len:
        pad_len = target_len - curr_len
        pose_data = np.pad(pose_data, pad_width=((0, 0), (0, pad_len), (0, 0)))
        valid_mask = np.concatenate(
            [np.ones(valid_len, dtype=np.float32), np.zeros(pad_len, dtype=np.float32)]
        )
    else:
        valid_mask = np.ones(target_len, dtype=np.float32)

    return pose_data, valid_mask


def process_input_data(input_path, config):
    """处理输入数据"""
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"❌ 输入文件不存在: {input_path}")
        exit(1)
    
    # 提取文件名
    input_filename = os.path.basename(input_path)
    
    # 只接受npy文件
    if not input_path.endswith('.npy'):
        print(f"❌ 只支持HumanML3D格式的.npy文件: {input_path}")
        exit(1)
    
    # 加载npy文件 (HumanML3D格式)
    try:
        pose_data = np.load(input_path)
        print(f"✅ 加载HumanML3D格式文件: {input_filename}")
    except Exception as e:
        print(f"❌ 加载npy文件失败: {e}")
        print("请确保输入文件是有效的npy文件")
        exit(1)
    
    # 检查数据维度
    if not hasattr(pose_data, 'ndim'):
        print(f"❌ 输入数据格式不正确: {type(pose_data)}")
        print("期望的输入格式: numpy数组或包含'pose'键的字典")
        exit(1)
    
    if pose_data.ndim not in [2, 3]:
        print(f"❌ 输入数据维度不正确: {pose_data.ndim}")
        print("期望的输入维度: 2 (n_frames, 263) 或 3 (n_augmentations, n_frames, 263)")
        exit(1)
    
    # 处理数据维度
    # momask期望的输入格式是 (batch_size, n_frames, 263)
    if pose_data.ndim == 2:
        # 如果是 (n_frames, 263)，添加batch维度
        pose_data = np.expand_dims(pose_data, axis=0)
    elif pose_data.ndim == 3:
        # 如果是 (n_augmentations, n_frames, 263)，取第一个增强版本
        pose_data = pose_data[0:1, ...]
    
    # 与训练时的MoMaskPreprocessor保持一致：默认取序列中间片段，不足时尾部补零并保留pad mask。
    target_len = int(config['source_seq_len'])
    pose_data, valid_mask_np = _clip_or_pad_sequence(
        pose_data,
        target_len=target_len,
        select_middle=bool(config.get('select_middle', True)),
    )
    
    # 归一化处理
    if config['data_type'] == 'humanML3D':
        norm_data_path = Path(config['humanML3D_normalization_data_path']) / f"{'LODO_' if config['LODO'] else ''}{config['dataset']}"
        if not norm_data_path.exists():
            print(f"❌ 归一化数据文件不存在: {norm_data_path}")
            print("请确保数据集已正确下载")
            exit(1)
        try:
            mean = np.load(norm_data_path / "Mean.npy")
            std = np.load(norm_data_path / "Std_FEAT_BIAS_included.npy")
            
            # 检查数据维度是否匹配
            if pose_data.shape[-1] != mean.shape[0]:
                print(f"⚠️ 数据维度不匹配: pose_data={pose_data.shape}, mean={mean.shape}")
                print(f"跳过归一化，使用原始数据")
                # 可以选择在这里添加其他处理方式
            else:
                pose_data = (pose_data - mean) / std
        except Exception as e:
            print(f"❌ 加载归一化数据失败: {e}")
            print("请确保归一化数据文件格式正确")
            exit(1)
    
    # 转换为张量
    pose_data = torch.tensor(pose_data, dtype=torch.float32).to(_DEVICE)
    
    # 创建metadata和valid_mask
    metadata = torch.tensor([[]], dtype=torch.float32).to(_DEVICE)  # 空metadata
    valid_mask = torch.from_numpy(valid_mask_np).unsqueeze(0).float().to(_DEVICE)
    
    return pose_data, metadata, valid_mask

def infer(model, pose_data, metadata, valid_mask, config):
    """进行推理"""
    with torch.no_grad():
        if config.get('medication', False):
            # 如果需要用药状态，默认为0（未用药）
            med = torch.tensor([0], dtype=torch.float32).to(_DEVICE)
            output = model(pose_data, metadata, med, valid_mask=valid_mask)
        else:
            output = model(pose_data, metadata, valid_mask=valid_mask)
    
    # 获取预测结果
    _, predicted_class = torch.max(output, 1)
    predicted_score = predicted_class.item()
    
    return predicted_score


def infer_with_details(model, pose_data, metadata, valid_mask, config):
    """Return score, logits and probabilities for UI/API usage."""
    with torch.no_grad():
        if config.get('medication', False):
            med = torch.tensor([0], dtype=torch.float32).to(_DEVICE)
            output = model(pose_data, metadata, med, valid_mask=valid_mask)
        else:
            output = model(pose_data, metadata, valid_mask=valid_mask)

    probabilities = torch.softmax(output, dim=1)
    predicted_score = torch.argmax(probabilities, dim=1).item()
    return {
        'predicted_updrs_score': int(predicted_score),
        'class_probabilities': probabilities.squeeze(0).detach().cpu().tolist(),
        'logits': output.squeeze(0).detach().cpu().tolist(),
    }


def run_inference(input_path, model_path, config_path='configs/momask/inference_config.json'):
    """Front-end friendly wrapper around the existing CLI pipeline."""
    config = load_config(config_path)
    model = load_model(config, model_path)
    pose_data, metadata, valid_mask = process_input_data(input_path, config)
    result = infer_with_details(model, pose_data, metadata, valid_mask, config)
    result.update({
        'input_file': input_path,
        'input_filename': os.path.basename(input_path),
        'model_path': model_path,
        'config_path': config_path,
        'device': str(_DEVICE),
        'num_classes': int(config.get('num_classes', len(result['class_probabilities']))),
        'dataset': config.get('dataset'),
        'backbone': config.get('backbone', 'momask')
    })
    return result

def main():
    parser = argparse.ArgumentParser(description='推理脚本，使用训练好的momask模型预测UPDRS评分')
    parser.add_argument('--config', type=str, default='configs/momask/inference_config.json', help='配置文件路径')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--input', type=str, required=True, help='输入骨架pkl文件路径')
    parser.add_argument('--output', type=str, default='prediction.json', help='输出预测结果文件路径')
    
    args = parser.parse_args()
    
    # 加载配置
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"❌ 加载配置文件失败: {e}")
        exit(1)
    
    # 加载模型
    print(f"🔍 加载模型: {args.model_path}")
    model = load_model(config, args.model_path)
    
    # 处理输入数据
    print(f"📊 处理输入数据: {args.input}")
    pose_data, metadata, valid_mask = process_input_data(args.input, config)
    
    # 进行推理
    print("🧠 进行推理...")
    predicted_score = infer(model, pose_data, metadata, valid_mask, config)
    
    # 提取输入文件名
    input_filename = os.path.basename(args.input)
    
    # 输出结果
    result = {
        'input_file': args.input,
        'input_filename': input_filename,
        'predicted_updrs_score': predicted_score
    }
    
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"❌ 保存结果失败: {e}")
        exit(1)
    
    print(f"✅ 预测完成！")
    print(f"📄 输入文件: {input_filename}")
    print(f"📊 预测评分: {predicted_score}")
    print(f"� 结果已保存到: {args.output}")

def cli_main():
    parser = argparse.ArgumentParser(
        description='Run UPDRS inference with detailed JSON output.'
    )
    parser.add_argument('--config', type=str, default='configs/momask/inference_config.json')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='prediction.json')
    args = parser.parse_args()

    try:
        result = run_inference(
            input_path=args.input,
            model_path=args.model_path,
            config_path=args.config
        )
    except Exception as e:
        print(f"Inference failed: {e}")
        raise SystemExit(1)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Input file: {result['input_filename']}")
    print(f"Predicted UPDRS score: {result['predicted_updrs_score']}")
    print(f"Class probabilities: {result['class_probabilities']}")
    print(f"Saved result to: {args.output}")


if __name__ == "__main__":
    cli_main()
