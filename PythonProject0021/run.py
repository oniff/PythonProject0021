import os
import torch  # 确保引入了 torch
from transformers import get_linear_schedule_with_warmup
from data_process import prepare_data
from model import SentimentClassifier
from train import train_epoch, eval_model

# ======================================================================
# 核心修改点：解开三模型注释，并全部将路径无缝指向你本地的 local_models 离线资产库
# ======================================================================
MODELS_TO_EXPERIMENT = {
    "BERT": "local_models/bert-base-chinese",
    "RoBERTa": "local_models/hfl/chinese-roberta-wwm-ext",
    "MacBERT": "local_models/hfl/chinese-macbert-base",
}

DATA_PATH = "data/hotel_data.csv"
SAVE_DIR = "bert_hotel_model"
BATCH_SIZE = 16
EPOCHS = 3
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_experiment():
    os.makedirs(SAVE_DIR, exist_ok=True)
    results = {}

    for model_name, model_path in MODELS_TO_EXPERIMENT.items():
        print(f"\n" + "=" * 50)
        print(f" 开始训练实验模型: {model_name} ({model_path})")
        print("=" * 50)

        try:
            # 加载数据
            train_loader, val_loader, _ = prepare_data(DATA_PATH, model_path, max_len=MAX_LEN, batch_size=BATCH_SIZE)

            # 初始化模型
            model = SentimentClassifier(model_path, num_classes=3).to(DEVICE)

            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
            total_steps = len(train_loader) * EPOCHS
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
            loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)

            best_accuracy = 0

            for epoch in range(EPOCHS):
                print(f"Epoch {epoch + 1}/{EPOCHS}")
                train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, DEVICE, scheduler)
                print(f"Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")

                val_acc, val_loss = eval_model(model, val_loader, loss_fn, DEVICE)
                print(f"Val loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")

                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    # 保存该模型的最优权重
                    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{model_name}_best.bin"))

            # 修复安全兜底：如果 best_accuracy 是 Tensor，转换为普通标量 float 保存到字典中
            if isinstance(best_accuracy, torch.Tensor):
                results[model_name] = best_accuracy.item()
            else:
                results[model_name] = best_accuracy

            print(f" {model_name} 训练完成！最佳准确率: {results[model_name]:.4f}")

        except Exception as e:
            print(f"模型 {model_name} 运行失败，错误信息: {e}")

    # 打印最终对比看板
    print("\n" + "==== 对比实验最终结果 ====")
    for name, score in results.items():
        print(f"模型: {name:<12} | 验证集最佳准确率: {score:.4f}")


if __name__ == "__main__":
    run_experiment()
