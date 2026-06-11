import jieba.analyse

def extract_keywords(text, top_k=5):
    """
    提取评论中的关键特征词
    """
    if not text.strip():
        return []
    # 使用 TextRank 算法提取名词和动词相关关键词更适合酒店场景
    keywords = jieba.analyse.textrank(text, topK=top_k, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v', 'a'))
    return keywords

if __name__ == "__main__":
    print(extract_keywords("房间干净整洁，服务热情周到，非常满意"))
