import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


class HotelDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_len=128):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        label = self.labels[item]

        encoding = self.tokenizer(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(label, dtype=torch.long)
        }


def prepare_data(data_path, model_name_or_path, max_len=128, batch_size=16, test_size=0.2):
    df = pd.read_csv(data_path)

    # ===================== 【数据清洗安全防线】 =====================
    # 1. 过滤掉可能因为多次写入或错位导致的无用表头行（核心过滤逻辑）
    df = df[df['label'] != 'label']
    df = df[df['review'] != 'review']

    # 2. 标签文本与数字映射字典
    label_mapping = {"负面": 0, "差评": 0, "负": 0, "不好": 0,
                     "中性": 1, "中评": 1, "一般": 1,
                     "正面": 2, "好评": 2, "正": 2, "好": 2}

    # 3. 执行安全的标签映射与强制数值清洗
    if df['label'].dtype == object:
        # 将文本标签转为数字 0,1,2，无法转换的脏数据（比如空行）会被标记为 NaN
        df['label'] = df['label'].map(label_mapping)

    # 4. 兜底清洗：利用 pd.to_numeric 将残留的非数字行转化为 NaN
    df['label'] = pd.to_numeric(df['label'], errors='coerce')

    # 5. 坚决删掉包含空值（NaN）的行，保证送入模型的全是干净、合格、成对的数据
    df = df.dropna(subset=['label', 'review'])

    # 6. 此时可以 100% 安全地将全员转换为标准整数类型
    df['label'] = df['label'].astype(int)
    # =============================================================

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    df_train, df_val = train_test_split(df, test_size=test_size, random_state=42)

    train_dataset = HotelDataset(
        reviews=df_train.review.reset_index(drop=True).to_numpy(),
        labels=df_train.label.reset_index(drop=True).to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    val_dataset = HotelDataset(
        reviews=df_val.review.reset_index(drop=True).to_numpy(),
        labels=df_val.label.reset_index(drop=True).to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, tokenizer
