import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import jieba
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import base64
from io import BytesIO
from transformers import AutoTokenizer, AutoModel
import warnings
import tempfile
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

warnings.filterwarnings("ignore")

# ==========================================
# 绘图环境全局配置（暗黑系学术大盘配色）
# ==========================================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.switch_backend('Agg')

app = FastAPI(title="酒店评论情感分析与文本挖掘系统后端 API")

# ==========================================
# 工业级跨域配置：确保前后端分离架构丝滑通信
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许纯前端独立 HTML 跨域调用
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 学术规范：物理对齐你训练时用的自定义分类网络
# ==========================================
class SentimentClassifier(nn.Module):
    def __init__(self, model_path, num_classes=3):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        self.drop = nn.Dropout(p=0.3)
        # 严格保持变量命名为 out，完美解耦并贴合你训练出的 RoBERTa_best.bin
        self.out = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)


# ==========================================
# 核心算法资产闭环：秒读本地离线 99.55% RoBERTa 权重
# ==========================================
model_path = "local_models/hfl/chinese-roberta-wwm-ext"
trained_weight_path = "bert_hotel_model/RoBERTa_best.bin"  # 你的学术最佳成果

tokenizer = AutoTokenizer.from_pretrained(model_path)
# 实例化自定义网络，完全规避 Transformers 默认的 classifier 结构冲突
model = SentimentClassifier(model_path, num_classes=3)

# 核心注入锁死：用你跑出的学术最佳权重进行全网层级参数软咬合
if os.path.exists(trained_weight_path):
    print(f"==================================================")
    print(f"🌟 正在成功注入微调后的 99.55% 核心权重: {trained_weight_path}")
    print(f"==================================================")
    model.load_state_dict(torch.load(trained_weight_path, map_location=torch.device('cpu')))
else:
    print(f"⚠️ 警告: 未在 {trained_weight_path} 路径下找到微调权重，当前分类头为随机初始化！")

model.eval()

# 维持一个全局内存 Dataframe 句柄，保障数据大盘初始化与多流向文件导出的动态一致性
global_active_df = pd.read_csv("data/hotel_data.csv")


def predict_texts(texts):
    """底层神经网络前向传播推理（完美适配自定义模型结构与张量矩阵）"""
    processed = [str(t).strip() for t in texts if len(str(t).strip()) > 0]
    if len(processed) == 0:
        return []

    inputs = tokenizer(processed, truncation=True, padding=True, max_length=128, return_tensors="pt")

    with torch.no_grad():
        # 显式传递解耦参数矩阵，提取未归一化的情感分类 Logits 得数空间
        logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

    preds = np.argmax(logits.numpy(), axis=1)
    return preds.tolist()


def generate_wordcloud(text):
    """TextRank 复合分词与全局高频词云拓扑图绘制"""
    words = jieba.lcut(text)
    words = [w for w in words if len(w) > 1]
    if len(words) == 0:
        words = ["无有效文本"]
    # 词云图底色融于前端整体的暗黑系大盘样式 (#252941)
    wc = WordCloud(width=800, height=400, background_color="#252941", font_path="C:/Windows/Fonts/msyh.ttc")
    wc.generate(" ".join(words))
    buf = BytesIO()
    wc.to_image().save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def generate_visuals(df):
    """动态捕获当前大盘矩阵，生成多维度可视化图表流"""
    df = df.copy()
    if "预测标签" in df.columns:
        df["label"] = df["预测标签"]
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(1).clip(0, 2)
    cnt = df["label"].value_counts().sort_index()
    labels = ["负面", "中性", "正面"]
    values = cnt.reindex([0, 1, 2], fill_value=0).tolist()
    neg_num, neu_num, pos_num = values
    total = sum(values)

    # 1. 情感分布饼图
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    ax1.pie(values, labels=labels, autopct="%1.1f%%", colors=["#ff4444", "#ffbb33", "#00C851"],
            textprops={'color': "white"})
    ax1.set_title("情感分布", color="white", fontsize=14, pad=10)
    fig1.patch.set_facecolor('#252941')
    buf1 = BytesIO()
    plt.savefig(buf1, format="png", bbox_inches="tight", facecolor='#252941')
    pie = base64.b64encode(buf1.getvalue()).decode()
    plt.close()

    # 2. 情感统计柱状图
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(labels, values, color=["#ff4444", "#ffbb33", "#00C851"])
    ax2.set_title("情感统计", color="white", fontsize=14, pad=10)
    ax2.set_facecolor('#252941')
    fig2.patch.set_facecolor('#252941')
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')
    for spine in ax2.spines.values():
        spine.set_color('white')
    buf2 = BytesIO()
    plt.savefig(buf2, format="png", bbox_inches="tight", facecolor='#252941')
    bar = base64.b64encode(buf2.getvalue()).decode()
    plt.close()

    all_text = " ".join(df.iloc[:, 0].astype(str).tolist())
    wc = generate_wordcloud(all_text)
    return pos_num, neu_num, neg_num, total, pie, bar, wc


# ======================================================================
# 【API 1】 路由路径：/api/init | 请求方法：GET | 功能：大盘看板基本面初始化
# ======================================================================
@app.get("/api/init")
def get_init_dashboard():
    global global_active_df
    pos_num, neu_num, neg_num, total, pie, bar, wc = generate_visuals(global_active_df)
    return {
        "pos_num": pos_num,
        "neu_num": neu_num,
        "neg_num": neg_num,
        "total": total,
        "pie_base64": pie,
        "bar_base64": bar,
        "wc_base64": wc
    }


# ======================================================================
# 【API 2】 路由路径：/api/analyze_single | 请求方法：POST | 功能：单条评论敏捷诊断
# ======================================================================
@app.post("/api/analyze_single")
def analyze_single(review: str = Form(...)):
    if not review.strip():
        raise HTTPException(status_code=400, detail="评论内容不能为空")

    # 激活本地推理链路
    preds = predict_texts([review])
    pred_label = preds[0]

    mapping = {0: "负面", 1: "中性", 2: "正面"}
    color_mapping = {0: "#ff4444", 1: "#ffbb33", 2: "#00C851"}

    return {
        "status": "success",
        "review": review,
        "label_id": pred_label,
        "sentiment": mapping[pred_label],
        "color": color_mapping[pred_label]
    }


# ======================================================================
# 【API 3】 路由路径：/api/upload_csv | 请求方法：POST | 功能：海量高吞吐 CSV 上传预测
# ======================================================================
@app.post("/api/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    global global_active_df
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="请上传标准格式的 CSV 文件")

    df = pd.read_csv(file.file)
    texts = df.iloc[:, 0].astype(str).tolist()
    preds = predict_texts(texts)

    df = df.iloc[:len(preds)].copy()
    df["预测标签"] = preds
    df["情感"] = df["预测标签"].map({0: "负面", 1: "中性", 2: "正面"})

    # 动态刷新全局缓存对象，保障多功能导出组件无缝对齐当前数据集
    global_active_df = df.copy()

    pos_num, neu_num, neg_num, total, pie, bar, wc = generate_visuals(df)
    details = df.to_dict(orient="records")

    return {
        "pos_num": pos_num,
        "neu_num": neu_num,
        "neg_num": neg_num,
        "total": total,
        "pie_base64": pie,
        "bar_base64": bar,
        "wc_base64": wc,
        "details": details
    }


# ======================================================================
# 【API 4】 路由路径：/export/excel | 请求方法：GET | 功能：Excel 明细账单导出
# ======================================================================
@app.get("/export/excel")
def export_excel():
    global global_active_df
    df = global_active_df.copy()
    if "情感" not in df.columns:
        df["情感"] = df["label"].map({0: "负面", 1: "中性", 2: "正面"})
    path = "酒店评论情感统计.xlsx"
    df.to_excel(path, index=False)
    return FileResponse(path, filename="酒店评论情感统计.xlsx")


# ======================================================================
# 【API 5】 路由路径：/export/pdf | 请求方法：GET | 功能：PDF 矢量报告流式导出
# ======================================================================
@app.get("/export/pdf")
def export_pdf():
    global global_active_df
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    font_zh = "STSong-Light"

    df = global_active_df.copy()
    if "预测标签" in df.columns:
        df["label"] = df["预测标签"]
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(1).clip(0, 2)
    cnt = df["label"].value_counts().sort_index()
    neg, neu, pos = cnt.reindex([0, 1, 2], fill_value=0).tolist()
    total = pos + neu + neg

    pie_path = tempfile.mktemp(suffix=".png")
    bar_path = tempfile.mktemp(suffix=".png")
    wc_path = tempfile.mktemp(suffix=".png")

    fig1, ax1 = plt.subplots(figsize=(5, 5), dpi=120)
    ax1.pie([neg, neu, pos], labels=["负面", "中性", "正面"], autopct="%1.1f%%",
            colors=["#ff4444", "#ffbb33", "#00C851"])
    ax1.set_title("情感占比")
    plt.savefig(pie_path, bbox_inches="tight")
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=120)
    ax2.bar(["负面", "中性", "正面"], [neg, neu, pos], color=["#ff4444", "#ffbb33", "#00C851"])
    ax2.set_title("情感数量统计")
    plt.tight_layout()
    plt.savefig(bar_path, bbox_inches="tight")
    plt.close()

    wc_base64 = generate_wordcloud(" ".join(df.iloc[:, 0].astype(str)))
    with open(wc_path, "wb") as f:
        f.write(base64.b64decode(wc_base64))

    pdf_path = "酒店评论情感分析报告.pdf"
    c = canvas.Canvas(pdf_path, pagesize=A4)
    w, h = A4
    c.setFont(font_zh, 20)
    c.drawCentredString(w / 2, h - 60, "酒店评论情感分析可视化报告")
    c.setFont(font_zh, 14)
    c.drawString(80, h - 110, f"正面：{pos} 条")
    c.drawString(80, h - 140, f"中性：{neu} 条")
    c.drawString(80, h - 170, f"负面：{neg} 条")
    c.drawString(80, h - 200, f"总数：{total} 条")
    c.setFont(font_zh, 16)
    c.drawString(80, h - 240, "一、情感占比")
    Image(pie_path, 270, 270).drawOn(c, (w - 270) / 2, h - 530)
    c.showPage()
    c.setFont(font_zh, 16)
    c.drawString(80, h - 60, "二、情感数量统计")
    Image(bar_path, 340, 230).drawOn(c, (w - 340) / 2, h - 310)
    c.drawString(80, h - 350, "三、词云")
    Image(wc_path, 420, 210).drawOn(c, (w - 420) / 2, h - 580)
    c.save()
    return FileResponse(pdf_path, filename="酒店评论情感分析报告.pdf")
