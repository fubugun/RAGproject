## 基于 RAG 的本地知识库问答（Streamlit + 向量检索 + 大模型）。

### 功能概要
支持上传 PDF / TXT / MD，分块、向量化、持久化
余弦相似度 Top-K 检索 + 相似度阈值
检索结果拼进 Prompt，调用兼容 OpenAI 的 Chat API（如智谱）
可选：RAGAS 评测脚本

### 环境要求
Python 版本（3.10+）
需要能访问大模型 API；嵌入模型默认从 Hugging Face 拉取，国内可写 配置 HF_ENDPOINT 镜像

### 安装步骤
git clone https://github.com/fubugun/RAGproject.git
cd RAGproject
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt


### 配置
复制 env.example 为 .env
必填/常用变量：OPENAI_API_KEY、OPENAI_BASE_URL、OPENAI_CHAT_MODEL、EMBEDDING_MODEL、HF_ENDPOINT
勿将 .env 提交到 Git

### 运行方式
streamlit run streamlit_app.py

