# 将设计文档的png转成文字并进行向量化入库
import os
from datetime import datetime
from utils import uploader
import uuid
from sql_main import kgSQL
import json
from prompt_tmp import structure_json
from utils import get_embed,get_llm,clean_json_response
from milvus_main2 import insert_batch
from sql_main import insert_task, insert_task_node
from minio_main import uploadfile 
from openai import OpenAI
# 1. 创建数据库连接
db = kgSQL()
embed_url='http://192.168.3.251:8020/embed'
llm_url="http://192.168.3.251:5001/v1/"
api_key="sk-36a1237576ec42eaba3ce9208e2c0a0e"
llm_model="Qwen/Qwen3.5-27b-GPTQ-Int4" # Qwen3.5-35B-A3B
# 图纸名对应的标签
label={
    "设计/说明/详图":"designDescription",
    "平面图":"flowplan",
    "系统图":"system"
}

def png_to_minio(filepath,bucket='test2'): #(file_path, bucket, is_public=True):
    res=uploadfile(filepath,bucket)
    return res
# 通过qianwen3.5视觉语言大模型来识别设计说明的图纸内容，解析出设计说明信息，文本+表格信息，必须语义对齐，用结构化json格式输出
def process_png_by_VLM(png_url,task_id,use_id,label='designDescription'):
    #
    node_name="process_png_by_VLM_node"
    now_time = datetime.now() #.strftime("%Y-%m-%d %H:%M:%S")
    # 调用LLM服务,# minio url'http://192.168.3.251:9000/test2/test5.png'#png_url=#split_dxf_data[label]
    prompt=f"""
    # Role
    你是一个专业的文档 OCR 与数据结构化专家，擅长处理双栏排版的工程图纸说明。
    # Task
    请分析提供的设计说明图片，提取其中的所有文本和表格内容,必须保持源文档的语义顺序，并严格按照指定的 JSON 格式输出。
    # Output Format
    请输出纯文本的 JSON 数据，不要包含markdown代码块标记（如 ```json）,JSON 结构必须严格遵循以下定义：{structure_json}
    """
    try:
        client = OpenAI(
            api_key=api_key, 
            base_url=llm_url,   #"https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        completion = client.chat.completions.create(
            model=llm_model, 
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url","image_url": {"url":png_url}},
                        {"type": "text", "text":prompt},
                    ],
                },
            ],
        )
        print('解析的结果数据：',type(completion.choices[0].message.content),completion.choices[0].message.content)
        out_data = completion.choices[0].message.content
        # 需要对大模型的结果劲歌格式清洗，只保留json的格式
        # 解析JSON
        try:# 先尝试直接解析
            structured_data = json.loads(out_data)
        except Exception as e:
            print("llm result is not json str,====================",out_data[:20],e)
            structured_data = clean_json_response(out_data) # 调用清洗函数
        return structured_data # json.loads(out_data)
    except Exception as e:
        print(f"错误信息：{e}")
        return {"success":False,"json":""}

# 将解析出来的 s的服务，embed的服务
def embeddings_to_knowledge(structure_json,task_id,use_id,collection_name='testmilvus',alias_name="default",env=''):
    node_name="embeddings_to_knowledge_node"
    now_time = datetime.now()
    doc_id=str(uuid.uuid4())
    task_node_id=9999
    i=0 # 统计段落数量
    vc_data=[]
    for key,value in structure_json.items():
        i=i+1
        if "table" in key:  # 需要将表格的字典形式转换为str
            print("table:",type(value),value)
            value=json.dumps(value,ensure_ascii=False)  #else: #if "para" in key:

        #print("文字：",key,value[:5])  # 对于标题和文档不处理直接向量化
        if isinstance(value,list):
            print("key:value value is list",key)
            value=' '.join(value)

        emb_res=get_embed(embed_url, value)
        if emb_res['success']:
            embeding=emb_res['data']['embeddings']  # 1维数组
            tmp_data={
                "id": f"{doc_id}_{str(i)}",  # uuid+id,id作为整篇文章的顺序编号
                "vector": embeding,
                "text": value,
                "task_id":task_id,
                "use_id": use_id,
                "doc_id":doc_id,
                "task_node_id": task_node_id
                }
            vc_data.append(tmp_data)
        else:
            print("有错误啊！",emb_res)

    print("向量长度：",len(vc_data))
    insert_batch(vc_data,collection_name,alias_name,env)
    cost_time= (datetime.now() - now_time).total_seconds() 
    print("花费时间：",cost_time)
    return doc_id

if __name__=="__main__":
    useid="test20260306"
    task_id=99999
    filep='/data/gonghuimin/agent/agent/data/output/2_设计施工说明 主要设备材料表.png'#'/data/gonghuimin/agent/test5.png' /data/gonghuimin/agent/image2.png
    now_time = datetime.now()
    res=png_to_minio(filep)
    print("1111:",res)
    urrl=res['url'] # 'http://192.168.3.251:9000/test2/test5.png' #
    llmout=process_png_by_VLM(urrl,task_id,useid)
    print("2222:",(datetime.now()-now_time).total_seconds() )
    embeddings_to_knowledge(llmout,task_id,useid)