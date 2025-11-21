# Induction-Hardening-ML
>
> 机器学习在轴类结构感应淬火过程预测中的应用

## 📅 关键时间节点

- **2026-01-07**: ⚠️ **最终截点** (纸质版随堂交，电子版 24:00 前发送)
- **2026-01-05**: 必须完成所有代码测试，开始汇总 PPT 和 Word。
- **2025-12-15**: 完成模型初步跑通。

## 📖 项目简介

本项目为《机器学习基础理论及其在工程科学中的应用》课程大作业。
**选题方向：** [材料本构建模 / 轴类结构感应淬火预测]
**主要目标：**

1. 针对 [具体材料/场景] 进行数据分析与处理。
2. 构建 [具体模型，如神经网络/SVM] 模型。
3. 实现对 [应力/温度场/相变场] 的精确预测。

## 🚀 环境配置 (极速版)

本项目使用 **[uv](https://github.com/astral-sh/uv)** 进行依赖管理，确保所有成员环境完全一致。请务必按照以下步骤操作，**不要使用传统的 pip install**。

### 1. 安装 uv

- **WSL / macOS / Linux**:

  ```bash
  curl -lsSf https://astral.sh/uv/install.sh | sh
  ```

其他安装方式请参考其[官方文档](https://docs.astral.sh/uv/getting-started/installation/)

### 2. 克隆项目 & 同步环境

```bash
# 1. 克隆代码并进入项目目录
git clone https://github.com/Flower101010/Induction-Hardening-ML.git
cd Induction-Hardening-ML

# 2. 一键安装所有依赖 (Python + 库)
# uv 会根据 uv.lock 自动构建虚拟环境，无需手动配置
uv sync
```

### 3. 验证环境

```bash
# 运行测试命令
uv run python -c "import torch; print('环境配置成功！PyTorch版本:', torch.__version__)"
```

---

## 🤝 协作规范 (必读)

1. **分支管理**：
   - `main` 分支：仅存放**可运行、无报错**的稳定代码。
   - 个人开发：请新建分支 `dev-姓名` (例如 `dev-flos`)，开发完成后发起 Pull Request 合并到 main。
2. **文件提交**：
   - **严禁上传** 数据文件 (`.csv`, `.xlsx`) 和 大型模型权重 (`.pth` > 100MB)。
   - **Notebook**：提交前请 Clear Output，避免冲突。
3. **依赖管理 (uv)**：
   - 本项目严格统一环境。如果你需要引入新的 Python 包（例如 `scikit-learn`），**严禁**使用 `pip install`。
   - **正确做法**：

   ```bash
   uv add scikit-learn
   ```

   - 安装完成后，**必须**将更新后的 `pyproject.toml` 和 `uv.lock` 文件提交到 Git，以便其他组员同步。

4. **Jupyter Notebook**：
   - 请使用 `uv run jupyter lab` 启动，确保使用正确环境。
   - **仅限实验与可视化**
      - Notebook (`.ipynb`) 仅用于数据探索、画图和简单验证。
      - **严禁**在 Notebook 中定义复杂的类（Model）或核心函数。
      - **正确做法**：将核心逻辑写在 `src/*.py` 中，然后在 Notebook 里 import 调用。
   - **提交前必清空 (Clear Output)**
      - `.ipynb` 文件包含大量的 JSON 格式输出（尤其是图片编码），极易导致 Git 冲突且无法解决。
      - **操作要求**：在 Commit 代码前，必须点击 `Edit` -> `Clear All Outputs`，保存后再提交。
      - *（未清空输出的 Notebook 将不被接受合并）*
   - **正确引用 src 模块**
      - 为了在 `notebooks/` 目录下的文件里顺利调用 `src/` 下的代码，请在所有 Notebook 的**第一个单元格**加入以下代码：

      ```python
      import sys
      import os
      
      # 将项目根目录加入路径，确保能 import src
      project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
      if project_root not in sys.path:
         sys.path.append(project_root)
      
      # 自动重载模块（修改 src 代码后无需重启 kernel）
      %load_ext autoreload
      %autoreload 2
      ```

## 📂 项目结构说明

```text
├── config/              # 配置文件
├── data/                # ⚠️ 数据文件 (已忽略，不要上传到 Git)
│   ├── raw/             # 请将原始 csv/xlsx 文件放入此处
│   └── processed/       # 代码生成的清洗后数据
├── docs/                # 文档
│   ├── references/      # 参考文献
│   └── report_drafts/   # 报告草稿
├── notebooks/           # 个人实验区 (命名格式：姓名_实验内容.ipynb)
├── outputs/             # 结果输出 (模型权重、图片)
│   ├── figures/         # 图片输出
│   ├── logs/            # 日志文件
│   └── models/          # 模型权重
├── scripts/             # 辅助脚本
├── src/                 # 核心代码库
├── tests/               # 测试代码
├── main.py              # 主程序入口
├── pyproject.toml       # 依赖配置文件
└── uv.lock              # 环境版本锁定文件
```

## 💻 如何运行代码

为了满足作业“代码可复现”的要求，请统一使用以下命令运行：

*由于目前项目仍在开发中，以下为设想的基础运行流程，后续会根据需要进行更改补充。*

### 1. 准备数据

由于数据文件较大，请从[课程网站上]下载 `dataset.zip`，解压后将文件放入 `data/raw/` 目录。

### 2. 训练模型

```bash
uv run scripts/run_train.py
```

### 3. 预测与评估

```bash
uv run scripts/run_evaluate.py
```

### 4. 启动 Jupyter Notebook (用于实验)

```bash
uv run jupyter lab
```

## 📊 任务分工 (Draft)

*作业要求明确成员分工及工作量占比，请大家在此处实时更新自己的工作内容。*

下面是示例：

| 成员 | 主要职责 | 当前任务 | 预计工作量占比 |
| :--- | :--- | :--- | :--- |
| **[Flos]** | 统筹、架构搭建、Pipeline整合 | 初始化项目、编写训练脚本 | TBD |
| **[组员A]** | 数据工程 | 数据清洗、特征提取 | TBD |
| **[组员B]** | 模型构建 | 搭建 Baseline 模型 | TBD |
| **[组员C]** | 调参优化 | 尝试 Transformer/LSTM | TBD |
| **[组员D]** | 可视化与报告 | TBD | TBD |
