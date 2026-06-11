import torch
from transformers import AutoTokenizer
from model import SentimentClassifier


def predict_sentiment(text, model_name="MacBERT", model_path="hfl/chinese-macbert-base", max_len=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 重新加载结构并载入训练好的权重
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = SentimentClassifier(model_path, num_classes=3)
    model.load_state_dict(torch.load(f"bert_hotel_model/{model_name}_best.bin", map_location=device))
    model = model.to(device)
    model.eval()

    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, prediction = torch.max(outputs, dim=1)

    return prediction.item()


if __name__ == "__main__":
    # 测试一下
    test_text = "酒店前台态度很差，房间里还有蚊子，避雷！"
    # 这里假设对比实验后 MacBERT 效果最好
    pred = predict_sentiment(test_text, model_name="MacBERT", model_path="hfl/chinese-macbert-base")
    print(f"预测文本: {test_text}\n预测标签: {pred}")
