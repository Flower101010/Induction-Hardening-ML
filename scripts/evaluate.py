"""
Evaluation Script
=================
评估脚本
=================

This script is responsible for evaluating the trained FNO model on the test dataset.
此脚本负责在测试数据集上评估训练好的 FNO 模型。
It loads the trained model weights, processes the test data, and computes performance metrics
它加载训练好的模型权重，处理测试数据，并计算性能指标
(e.g., MSE, MAE, relative error) for temperature and phase transformation predictions.
（例如，MSE、MAE、相对误差）用于温度和相变预测。

Usage:
    uv run scripts/evaluate.py --config configs/model_config.yaml --checkpoint outputs/models_weights/best_model.pth
"""
