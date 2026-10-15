# 配置信息
import os

# llm info
llm_url = 'http://192.168.31.203:11434/v1'
api_key = "local"
llm_model ="qwen3:8b"
MAX_TOKENS = 8192
TEMPERATURE = 0.1
TOP_P = 0.3

#minio，IP 和端口, hostname：minio
minio_config={
    "endpoint":"192.168.31.203:9000", 
    "access_key":"admin",
    "secret_key":"xxxxxxxx",
    "bucket_name":"rag"  # 可以设置默认的
}
bucket = "test"  # 文件库名 "test2" 

# Milvus配置
MILVUS_CONFIG = {
'host': '192.168.3.251',#'58.211.165.130', # Milvus服务地址192.168.3.251
'port': 19530, # Milvus服务端口 'collection_name': 'doc_chunks_bm25',
'dim': 1024, # bge-m3模型的向量维度
'user':'root',
'password':'xxxxxxx'
}
collection_name = 'mykg2'   # 集合名,专业知识库
alias_name = "default"           # 集合库名

# mysql配置
MYSQL_CONFIG = {
    'host': 'r192.168.3.251',
    'database': 'new_kg', 
    'user': 'maxi_dev',
    'password':"xxxx",
    'port': '5307'
}

# 向量模型
bge_model={
    "path":"/data/embmodel/bge-large-zh-v1.5",
    "dim":1024
}
embed_url='http://192.168.31.201:9051/embed'  # 部署成api服务


