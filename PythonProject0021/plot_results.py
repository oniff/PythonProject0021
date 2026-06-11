import matplotlib.pyplot as plt

# 设置绘图风格，确保支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 严格提取自你控制台日志的真实实验数据
epochs = [1, 2, 3]

# 1. 验证集准确率数据 (Val Accuracy)
bert_acc = [0.9922, 0.9922, 0.9922]
roberta_acc = [0.9922, 0.9944, 0.9955]
macbert_acc = [0.9922, 0.9911, 0.9933]

# 2. 训练集损失函数数据 (Train Loss)
bert_loss = [0.1100, 0.0153, 0.0096]
roberta_loss = [0.1175, 0.0130, 0.0070]
macbert_loss = [0.1395, 0.0145, 0.0080]

# --- 绘图 1：准确率对比折线图 ---
plt.figure(figsize=(7, 5), dpi=300)
plt.plot(epochs, bert_acc, marker='o', linestyle='--', color='#7F7F7F', linewidth=2, label='BERT (99.22%)')
plt.plot(epochs, roberta_acc, marker='s', linestyle='-', color='#C5221F', linewidth=2.5, label='RoBERTa-wwm-ext (99.55%)')
plt.plot(epochs, macbert_acc, marker='^', linestyle='-.', color='#137333', linewidth=2, label='MacBERT (99.33%)')

plt.title("图 3-3 三款预训练模型验证集准确率（Accuracy）演进曲线", fontsize=11, weight='bold', pad=12)
plt.xlabel("迭代轮数 (Epoch)", fontsize=10)
plt.ylabel("准确率 (Accuracy)", fontsize=10)
plt.xticks(epochs)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig("accuracy_curve.png", dpi=300)
plt.close()

# --- 绘图 2：损失函数收敛图 ---
plt.figure(figsize=(7, 5), dpi=300)
plt.plot(epochs, bert_loss, marker='o', linestyle='--', color='#7F7F7F', linewidth=2, label='BERT')
plt.plot(epochs, roberta_loss, marker='s', linestyle='-', color='#C5221F', linewidth=2.5, label='RoBERTa-wwm-ext')
plt.plot(epochs, macbert_loss, marker='^', linestyle='-.', color='#137333', linewidth=2, label='MacBERT')

plt.title("图 3-4 三款预训练模型训练集损失函数（Loss）收敛曲线", fontsize=11, weight='bold', pad=12)
plt.xlabel("迭代轮数 (Epoch)", fontsize=10)
plt.ylabel("损失值 (Loss)", fontsize=10)
plt.xticks(epochs)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=300)
plt.close()

print("✅ 学术图表已成功生成：'accuracy_curve.png' 和 'loss_curve.png'，可直接插入论文！")
