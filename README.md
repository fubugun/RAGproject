## 基于 RAG 的本地知识库问答（Streamlit + 向量检索 + 大模型）。
本地文档上传 → 分块与向量化 → 相似度检索 → 大模型生成回答。Web 界面基于 Streamlit。

## 功能概要
支持上传 PDF / TXT / MD，分块、向量化、持久化
余弦相似度 Top-K 检索 + 相似度阈值
检索结果拼进 Prompt，调用兼容 OpenAI 的 Chat API（如智谱）










---

## 系统架构图

下图描述数据流与模块关系
<img width="3493" height="925" alt="exported_image (1)" src="https://github.com/user-attachments/assets/6bc73060-7fc0-473b-a856-685c85b80bda" />

用户上传文档后，经解析与分块，由 Sentence-Transformer 得到归一化向量并写入本地向量库；提问时同样向量化，用矩阵乘法得到与各块的相似度，经阈值过滤后取 Top-K 片段拼入提示词，调用兼容 OpenAI 协议的 Chat 接口返回回答。

---

## 技术栈说明

| 类别 | 技术 | 说明 |
|------|------|------|
| 语言 | Python 3.9+（推荐 3.10+） | 3.9.7 环境下 Streamlit 版本受限，界面为表单式问答 |
| Web | Streamlit | 单页应用：侧栏建库与参数，主区对话 |
| 嵌入 | sentence-transformers | 默认 `all-MiniLM-L6-v2`，可在 `.env` 中更换 |
| 向量检索 | NumPy | 向量 L2 归一化后，点积等价余弦相似度；Top-K + 阈值 |
| 大模型 | OpenAI 兼容 API | 如智谱 `OPENAI_BASE_URL` + `OPENAI_CHAT_MODEL` |
| PDF | pypdf | 文本抽取 |
| 配置 | python-dotenv | 从 `.env` 加载密钥与模型名 |
| 可选评测 | RAGAS + datasets | `requirements-ragas.txt`，脚本 `scripts/evaluate_ragas.py` |

---

## 数据来源说明

| 来源 | 用途 |
|------|------|
| **用户上传** | 通过网页上传的 PDF / TXT / Markdown，为系统主要知识来源；向量与元数据持久化在 `data/vector_store/`（默认已 `.gitignore`，不进入版本库） |
| **`data/eval_samples.jsonl`** | 可选：供 RAGAS 脚本使用的示例问答对（`question` / `ground_truth`），与主界面知识库无强制绑定，可自行替换为与自建库一致的问题 |

**注意**：请勿将含隐私或未授权文本的库推送到公开仓库；提交代码前确认 `.env` 未被跟踪。

---

## Demo 截图

上传文件+问答
<img width="2489" height="1358" alt="屏幕截图 2026-03-22 182408" src="https://github.com/user-attachments/assets/98192b21-59f7-4cdf-8755-ec9b924d28e1" />
<img width="2458" height="1249" alt="屏幕截图 2026-03-22 182631" src="https://github.com/user-attachments/assets/502644be-c83d-4978-a5dc-f74ec3f07e28" />
<img width="2477" height="1286" alt="屏幕截图 2026-03-22 182659" src="https://github.com/user-attachments/assets/9702266a-e628-4d99-af37-ac5a55a7040f" />



RAG评测

<img width="534" height="375" alt="屏幕截图 2026-03-22 182729" src="https://github.com/user-attachments/assets/f8ecd344-53da-4b39-9421-6a5c5893cd78" />
<img width="1081" height="697" alt="屏幕截图 2026-03-22 183955" src="https://github.com/user-attachments/assets/384f336e-df3d-4efd-b0c0-b41696f2ad1b" />




---

## 测试问题示例

上传本地文档“复试win.md”

| 序号 | 测试问题 | 预期行为 |
|------|----------|----------|
| 1 | memset| 助手：memset函数是C语言标准库中的函数，用于将内存块中的字节全部设置为某个指定的值。以下是memset函数的相关信息...|
| 2 | string库 | C++ 中的 string 库 是用来处理字符串（文本）的，非常常用 它来自标准库：#include <string>...|
| 3 | z字型打印数组| C++ 实现 Z 字型打印（Zigzag Conversion） 是一个经典题（例如 Zigzag Conversion）。题目大意：给一个字符串，把字符按 Z 字型排列，然后按行读取... |



---

## 环境要求
Python 版本（3.10+）
需要能访问大模型 API；嵌入模型默认从 Hugging Face 拉取，国内可写 配置 HF_ENDPOINT 镜像



## 配置
复制 env.example 为 .env
必填/常用变量：OPENAI_API_KEY、OPENAI_BASE_URL、OPENAI_CHAT_MODEL、EMBEDDING_MODEL、HF_ENDPOINT
勿将 .env 提交到 Git

## 安装与运行

```bash
git clone https://github.com/fubugun/RAGproject.git
cd RAGproject
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp env.example .env   # 复制后编辑 .env，填入 API 等
streamlit run streamlit_app.py
```

国内若无法访问 Hugging Face，可在 `.env` 中配置 `HF_ENDPOINT`（见 `env.example`）。

---

