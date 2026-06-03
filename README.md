# PV-IQA 使用说明

PV-IQA 用于掌静脉 ROI 图像质量评估。项目流程：

```text
训练识别模型 -> 提取 embedding -> 生成质量伪标签 -> 训练 IQA 模型 -> 图片质量打分
```

当前推荐使用的新权重：

```text
checkpoints\new_old_roi_c7c8c9_5_retrain\iqa\best.pt
```

## 1. 环境安装

推荐使用 Conda：

```powershell
conda create -n palm_iqa python=3.10 -y            #创建虚拟环境
conda activate palm_iqa                            #激活虚拟环境
cd D:\IQA\code\PV-IQA-project                      #调至项目目录
pip install -r requirements.txt                    #安装项目所需软件包
```

如果需要 GPU，请根据目标机器 CUDA 版本安装对应的 PyTorch CUDA 包。

## 2. 数据格式

训练数据按身份/左右手分文件夹：

```text
dataset_root/
├── ac_l/
│   ├── 1.jpg
│   └── 2.jpg
├── ac_r/
├── bhh_l/
└── bhh_r/
```
默认左右手分开作为不同类别。

## 3. 重新训练

完整训练：

```powershell
conda activate palm_iqa
cd D:\IQA\code\PV-IQA-project
python train_project.py --data "D:\IQA\data_base_path\IQA\New_old_roi_c7c8c9_5_numeric_only" --name new_old_roi_c7c8c9_5_retrain --device auto
```

快速试跑：

```powershell
python train_project.py --data "D:\IQA\data_base_path\IQA\New_old_roi_c7c8c9_5_numeric_only" --name debug_run --recog-epochs 2 --iqa-epochs 2 --device auto
```

训练输出：

```text
checkpoints\<run_name>\recognizer\best.pt
checkpoints\<run_name>\pseudo_labels\pseudo_labels.csv
checkpoints\<run_name>\iqa\best.pt
```

最终用于质量打分的是：

```text
checkpoints\<run_name>\iqa\best.pt
```

## 4. 测试打分

使用已有的权重给文件夹打分：

```powershell
conda activate palm_iqa
cd D:\IQA\code\PV-IQA-project    #来到项目目录，激活环境
python score_images.py --ckpt "checkpoints\new_old_roi_c7c8c9_5_retrain\iqa\best.pt" --input "D:\IQA\code\yinxintestdata\err_roi" --out "err_roi_scores_new_retrain.csv"
``
#"D:\IQA\code\yinxintestdata\err_roi"为想要测试的图片文件夹
#"checkpoints\new_old_roi_c7c8c9_5_retrain\iqa\best.pt"为目前已经训练好的权重

打分并按质量分重命名：

```powershell
python score_images.py --ckpt "checkpoints\new_old_roi_c7c8c9_5_retrain\iqa\best.pt" --input "D:\IQA\code\yinxintestdata\err_roi" --out "err_roi_scores_new_retrain.csv" --rename
```


单张图片：

```powershell
python score_images.py --ckpt "checkpoints\new_old_roi_c7c8c9_5_retrain\iqa\best.pt" --input "D:\IQA\code\yinxintestdata\err_roi\example.jpg" --out "one_image_score.csv"
```

## 5. 脚本参数

训练脚本：

```powershell
python train_project.py --help
```

常用参数：

```text
--data             训练数据目录
--name             实验名
--device           auto / cuda / cpu
--recog-epochs     识别模型训练轮数
--iqa-epochs       质量模型训练轮数
--skip-onnx        跳过 ONNX 导出
```

测试脚本：

```powershell
python score_images.py --help
```

常用参数：

```text
--ckpt       IQA 权重
--input      图片或文件夹
--out        输出 CSV
--rename     按质量分重命名
--digits     重命名保留小数位，默认 3
```

## 6. 常见问题

### base 环境报错

如果看到：

```text
TypeError: unsupported operand type(s) for |: 'type' and 'type'
```

说明 Python 版本太旧。切换环境：

```powershell
conda activate palm_iqa
```

### 训练很慢

检查是否用上 GPU：

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

