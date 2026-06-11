const API_BASE = "http://127.0.0.1:8000";

// 1. 单条分析
document.getElementById("btnAnalyze").addEventListener("click", async () => {
    const text = document.getElementById("singleReview").value.trim();
    if(!text) return alert("请输入内容");

    const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({review: text})
    });
    const data = await res.json();
    document.getElementById("singleResult").innerText = `分析结果：【${data.sentiment_text}】 | 关键词：${data.keywords.join(', ')}`;
});

// 2. 批量CSV上传与看板更新
document.getElementById("btnUpload").addEventListener("click", async () => {
    const fileInput = document.getElementById("csvFile");
    if(fileInput.files.length === 0) return alert("请选择CSV文件");

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    // 显示加载提示
    document.getElementById("btnUpload").innerText = "正在模型计算中...";

    try {
        const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: formData });
        const data = await res.json();

        // 渲染数字面板
        document.getElementById("posNum").innerText = data.stats.pos;
        document.getElementById("neuNum").innerText = data.stats.neu;
        document.getElementById("negNum").innerText = data.stats.neg;
        document.getElementById("totalNum").innerText = data.stats.total;

        // 渲染图片
        document.getElementById("imgPie").src = `data:image/png;base64,${data.stats.pie}`;
        document.getElementById("imgBar").src = `data:image/png;base64,${data.stats.bar}`;
        document.getElementById("imgWc").src = `data:image/png;base64,${data.stats.wordcloud}`;

        // 渲染明细表格
        const tbody = document.querySelector("#resultTable tbody");
        tbody.innerHTML = "";
        data.details.forEach(item => {
            const tr = document.createElement("tr");
            tr.innerHTML = `<td style="padding:10px">${item.review || item.评论内容}</td>
                            <td style="padding:10px; color:${item.情感==='正面'?'#00C851':item.情感==='负面'?'#ff4444':'#ffbb33'}">${item.情感}</td>`;
            tbody.appendChild(tr);
        });

        // 展示大盘
        document.getElementById("analysisDashboard").style.display = "block";
    } catch (e) {
        alert("上传分析失败，请检查后端运行状态。");
    } finally {
        document.getElementById("btnUpload").innerText = "上传并预测";
    }
});
