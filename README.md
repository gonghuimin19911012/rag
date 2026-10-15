# kg qa  知识问答
采用RAG混合检索

私有知识库 RAG 问答系统，实现文档解析 (PDF/DOCX)、图片图纸 VLM 解析、文档分块、BGE 向量化、Milvus 混合检索（向量检索 + BM25 稀疏检索 RRFRerank 融合）、Ollama 兼容大模型问答，支持普通非流式问答、SSE 流式问答，问答对话记录持久化到 MySQL 数据库，MinIO 用于图纸文件存储。

> 
> 核心技术栈：Python3.10+、FastAPI、Milvus、MySQL、Sentence‑Transformers (BGE‑Large‑ZH‑v1.5)、Ollama (OpenAI 兼容接口)、PyMuPDF、python‑docx、pymilvus

## 目录结构

```
.
├── config.py                # 全局配置：LLM、Milvus、MySQL、MinIO、Embedding模型配置
├── embedmodel_new.py        # BGE向量化封装，支持单条/批量，GPU轮询负载均衡
├── milvus_main.py           # Milvus操作封装：集合创建、增删改查、hybrid_search混合检索
├── process_data_milvus.py   # PDF/DOCX文档解析、清洗、分块、向量化入库主逻辑
├── process_png_to_json.py   # PNG工程图纸，VLM视觉大模型解析，结构化后入库
├── sql_main.py              # MySQL数据库封装，会话、消息、任务表CRUD
├── run_qa.py                # FastAPI 非流式RAG问答服务（端口9080）
├── run_qa_stream.py         # FastAPI SSE流式RAG问答服务（端口9081）
├── run_qa2.py               # 另一个非流式问答服务实现
├── prompt_tmp.py            # LLM提示词模板（图纸结构化解析使用）
├── requirements.txt         # 项目依赖包
└── README.md                # 项目说明文档
```

## 主要功能

1. **文档解析入库**
   - 支持 PDF、DOCX 工程文档自动解析，自动过滤页眉页脚、页码噪声；
   - 智能文本合并与切分，自动处理表格，表格超长自动拆分；
   - 自动 MD5 文档切片去重，避免重复入库 Milvus；
2. **图纸图片解析入库**
   - 工程 PNG 图纸，调用 VLM 视觉大模型提取图文 + 表格，输出结构化 JSON；
   - 解析后文本向量化存入 Milvus 知识库，图纸原图上传 MinIO 存储；
3. **向量库能力**
   - 使用`bge‑large‑zh‑v1.5`，向量维度 1024；
   - Milvus 混合检索：稠密向量 IP 内积检索 + BM25 稀疏检索，RRF 融合排序；
   - 支持集合自动创建、加载重试、数据插入、删除、更新；
4. **问答服务**
   - 普通非流式问答接口；
   - SSE 流式输出问答接口，适合前端页面打字机效果；
   - 多轮对话，内存缓存会话历史，自动过期清理；
   - 检索溯源返回来源文档片段、相关度分数；
   - 问答会话、消息记录持久化 MySQL；
5. **大模型对接**
   - 兼容 OpenAI 协议，对接 Ollama/Qwen 系列模型；
   - 文档无信息时，强制提示 “根据提供的文档，我无法找到相关信息”。

## 环境部署

### 1. 依赖安装

> 
> 生成 requirements.txt 参考

```
fastapi>=0.104.1
uvicorn[standard]>=0.24.0
pymilvus>=2.4.0
sentence-transformers>=2.7.0
torch>=2.1.0
mysql‑connector‑python>=8.3.0
pymupdf>=1.24.0
python‑docx>=1.1.2
sse‑starlette>=0.4.0
langchain‑openai>=0.1.5
langchain‑core>=0.1.20
openai>=1.12.0
requests>=2.31.0
numpy>=1.26.0
```

安装：

```
pip install -r requirements.txt
```

### 2. 环境准备（中间件）

表格

| 组件 | 说明 |
| --- | --- |
| Milvus | 向量数据库，版本 >=2.4，开启 BM25 Function 功能； |
| MySQL | 存储会话、消息、任务记录；提前创建数据库`new_kg`； |
| MinIO | 图纸 png 文件对象存储； |
| Ollama 服务 | 部署 qwen3:8b，提供 openai 兼容接口； |
| BGE 模型 | 提前下载`bge‑large‑zh‑v1.5`至服务器本地路径； |

### 3. 修改配置 config.py

> 
> 修改所有 IP、端口、账号密码、模型本地路径，适配你的环境

```
# LLM服务地址（Ollama）
llm_url = 'http://192.168.31.203:11434/v1'
api_key = "local"
llm_model ="qwen3:8b"

# MinIO配置
minio_config={
    "endpoint":"192.168.31.203:9000",
    "access_key":"admin",
    "secret_key":"xxxxxxxx",
    "bucket_name":"rag"
}

# Milvus向量库
MILVUS_CONFIG = {
'host': '192.168.3.251',
'port': 19530,
'dim': 1024,
'user':'root',
'password':'xxxxxxx'
}
collection_name = 'mykg2'   # milvus集合名称

# MySQL数据库
MYSQL_CONFIG = {
    'host': '192.168.3.251',
    'database': 'new_kg',
    'user': 'maxi_dev',
    'password':"xxxx",
    'port': '5307'
}

# Embedding模型本地路径
bge_model={
    "path":"/data/embmodel/bge-large-zh-v1.5",
    "dim":1024
}
# 可选：独立部署embedding api服务地址
embed_url='http://192.168.31.201:9051/embed'
```

### 4. MySQL 数据表准备

需要提前执行建表 SQL（sql_main.py 使用到三张核心表）

```
-- 会话主表 conversation_info
CREATE TABLE conversation_info (
  id VARCHAR(128) PRIMARY KEY COMMENT '会话id',
  create_time VARCHAR(32),
  use_id VARCHAR(128) COMMENT '用户id',
  task_id INT COMMENT '项目任务id',
  info TEXT COMMENT '首次提问内容',
  session_id VARCHAR(128),
  type TINYINT COMMENT '1:问数，2:专业知识问答，3：图纸问答'
);

-- 问答消息表 message_info
CREATE TABLE message_info (
  id VARCHAR(128) PRIMARY KEY,
  create_time VARCHAR(32),
  doc_id VARCHAR(128) COMMENT '关联会话id',
  task_id INT,
  info TEXT COMMENT 'json字符串，存储q/a',
  type TINYINT
);

-- 任务节点日志表 task_node_info
CREATE TABLE task_node_info (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  create_time VARCHAR(32),
  use_id VARCHAR(128),
  task_id INT,
  node_name_en VARCHAR(128),
  output TEXT,
  end_time VARCHAR(32),
  cost_time FLOAT
);

-- 任务信息表 task_info
CREATE TABLE task_info (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  create_time VARCHAR(32),
  use_id VARCHAR(128),
  flag TINYINT,
  file_name_list TEXT,
  file_minio_list TEXT,
  end_time VARCHAR(32)
);
```

## 启动服务

### 方式 1：非流式问答服务 run_qa.py

```
python run_qa.py
# 服务地址 http://0.0.0.0:9080
# 接口文档：http://ip:9080/docs
```

接口：

- `POST /api/qa/ask` 单次问答
- `POST /api/qa/chat` 多轮对话
- `GET /api/qa/search` 只做知识库检索，不调用大模型

### 方式 2：SSE 流式问答服务 run_qa_stream.py（推荐前端对接）

```
python run_qa_stream.py
# 服务地址 http://0.0.0.0:9081
```

接口：

- `POST /qa_ask` SSE 流式问答接口，返回`start/progress/sources/answer/done/error`事件流

### 文档入库脚本使用

1. PDF/DOCX 批量入库 `process_data_milvus.py`
修改代码中`root_dir`为你的文档存放目录，运行脚本自动解析、分块、向量化写入 Milvus

```
if __name__ == "__main__":
    root_dir="/data/gonghuimin/agent/kg_QA/data/over" #修改为你的文档目录
```

```
python process_data_milvus.py
```

2. PNG 图纸解析入库 `process_png_to_json.py`

```
python process_png_to_json.py
```

逻辑：本地 png 上传 MinIO → VLM 大模型图片 OCR 结构化解析 → 解析结果文本向量化存入 Milvus。

### embedding 模型独立服务

> 
> 项目提供两种 embedding 调用模式

1. 进程内加载 BGE 模型（embedmodel_new.py 本地 GPU 运行）
2. 独立 HTTP embedding 服务 `embed_url`，适合多服务共享模型，避免多进程重复加载占用显存。

## API 示例

### 流式问答 POST /qa_ask

请求体

```
{
    "user_question":"墙体排风设计要求是什么？",
    "project_id":1001,
    "user_id":"user_001",
    "conversation_id":"conv‑xxxx",
    "top_k":8
}
```

返回 SSE 事件流，event=message，data 为 json 字符串。

### 非流式问答 POST /api/qa/ask

```
{
  "task_id": 1001,
  "use_id": "user_001",
  "doc_id": null,
  "question": "墙体排风设计要求是什么？",
  "top_k": 8,
  "temperature":0.3,
  "max_tokens":4096
}
```

## 关键说明与注意事项

1. **Milvus 注意点**
   - 集合自动创建开启中文分析器，同时创建 FLAT 向量索引、SPARSE_WAND BM25 稀疏索引；
   - 集合加载增加重试逻辑；
   - 切片 MD5 哈希做文档去重；
   - hybrid_search 使用 RRF 算法融合稠密向量 + BM25 检索结果。
2. **GPU 显存**
   - embedmodel_new.py 实现多 worker 轮询分配 GPU，防止全部抢占 0 号卡 OOM；
   - 大模型 (Ollama) 和 Embedding 模型建议分开部署，减少显存冲突。
3. **MySQL 连接**
   - 数据库连接会自动检测断开重连；流式服务每次操作数据库创建新连接，执行完毕关闭，防止连接泄漏。
4. **会话存储**
   - run_qa_stream.py 会话保存在内存，服务重启会话丢失；生产环境建议迁移 Redis。
   - 设置会话 24 小时 TTL 自动清理过期内存缓存。
5. **文档分块策略**
   - 默认普通文本最大块 500 字符；表格最大 2000 字符，超长自动分割，保留表头。
6. **故障排查**
   - 向量化失败：检查 BGE 模型路径，或者独立 embed 服务是否正常；
   - Milvus 报错：确认集合存在，服务端口网络通，账号密码；
   - LLM 调用失败：确认 ollama 服务正常，模型已经 pull 下载完成。

## 后续优化建议（可选）

1. 会话历史迁移 Redis，支持多实例部署；
2. 增加文档管理 API：上传文件接口、删除知识库文档接口；
3. 增加向量库过滤，支持按项目、文档名称过滤检索；
4. 增加评估模块，RAG 问答效果评估；
5. 增加日志组件，日志落盘；
6. 增加接口鉴权 token；
7. 支持 word、txt、cad 图纸等更多格式解析。

## 版本信息

- 当前版本：V1.0
- 场景：建筑工程专业知识库 RAG 系统
- 模型：Qwen 系列 LLM + bge‑large‑zh‑v1.5 Embedding
