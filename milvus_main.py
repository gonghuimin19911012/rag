from pymilvus import connections,FieldSchema,CollectionSchema,DataType,Collection,utility
from pymilvus import Function, FunctionType
from pymilvus import MilvusClient
import hashlib
from config import MILVUS_CONFIG
from pymilvus import AnnSearchRequest, RRFRanker,MilvusClient #, WeightedRanker

token = f"{MILVUS_CONFIG['user']}:{MILVUS_CONFIG['password']}"
class myMilvus():
    def __init__(self,collection_name,alias_name="default",env=''): # 增加环境标识
        # 初始化Milvus连接和集合
        self.env=env.strip()
        self.alias=alias_name.strip()
        self.collection_name=collection_name.strip()
        self.init_milvus()
        #self.collection_name

    def init_milvus(self,):
        """初始化Milvus连接和集合"""
        #alias_name="default"
        try: # 链接milvus# 阿里云 Milvus 使用 token 认证
            #token = f"{MILVUS_CONFIG['user']}:{MILVUS_CONFIG['password']}"
            connections.connect(
                alias=self.alias,
                host=MILVUS_CONFIG['host'],
                port=MILVUS_CONFIG['port'],
                token=token
            )
            milvus_name=f"{self.env}{self.collection_name}" # 如果集合不存在则创建,需要区分不同开发环境，env
            print(f"当前使用的{self.alias} Milvus集合名：{milvus_name}") # 新增调试日志
            # 判断是否有集合，没有则创建
            if not utility.has_collection(milvus_name):
                # 创建并定义字段  定义集合 schema
                schema = MilvusClient.create_schema(auto_id=False,enable_dynamic_field=False)
                analyzer_params = {"type": "chinese"} # english
                schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=64,is_primary=True)
                schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=10240,enable_analyzer=True,analyzer_params=analyzer_params,enable_match=True) #, enable_match=True , analyzer_params=analyzer_params
                schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=MILVUS_CONFIG['dim'])
                schema.add_field(field_name="sparse_bm25", datatype=DataType.SPARSE_FLOAT_VECTOR,description="BM25 算法生成的稀疏向量")
                schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=64)
                schema.add_field(field_name="doc_name", datatype=DataType.VARCHAR, max_length=1024)
                schema.add_field(field_name="doc_hash", datatype=DataType.VARCHAR, max_length=32)
                bm25_func_name = "bm25_sparse"
                bm25_func = Function(
                    name=bm25_func_name,
                    function_type=FunctionType.BM25,
                    input_field_names=["text"], # 输入：text 字段
                    output_field_names="sparse_bm25" # 输出：新增的稀疏向量字段
                )
                schema.add_function(bm25_func) #create_function(bm25_func)
                print(f"已注册 BM25 函数 '{bm25_func_name}'")
                self.collection = Collection(name=milvus_name,schema=schema,using=self.alias)
                # 创建索引
                index_params = {
                    "index_type": "FLAT", # FLAT 无需配置 nlist，实现更简单，且小数据量下搜索速度可能比 IVF_FLAT 更快;仅当数据量接近 10 万级、且 FLAT 搜索速度无法满足需求时，再切换到 IVF_FLAT。
                    "metric_type": "IP", # 内积象是读适合文本，L2欧氏距离适合图像 "params": {"nlist": 128} # 若 nlist=128，nprobe 设为 10,查询时聚类数量
                    "params":{"nlist": 128} # FLAT无额外参数，空字典即可
                }
                index_params2 = {
                    "index_type": "SPARSE_WAND",
                    "metric_type": "BM25"
                }
                self.collection.create_index(field_name="vector", index_params=index_params)
                self.collection.create_index(field_name="sparse_bm25", index_params=index_params2)
                print(f"Milvus集合 {milvus_name} 创建成功=======================") 
            else:
                self.collection = Collection(name=milvus_name,using=self.alias) 
                print(f"成功加载Milvus集合 {milvus_name}========================") 
            # 加载集合到内存（带重试机制）
            self._load_collection_with_retry()
        except Exception as e:
            print(f"Milvus初始化错误: {e}")
            raise

    def _load_collection_with_retry(self, max_retries=3, wait_seconds=3):
        """带重试机制的集合加载"""
        from time import sleep
        for attempt in range(max_retries):
            try:
                self.collection.load()
                self.collection.flush()
                print(f"集合 {self.collection.name} 加载成功")
                return
            except Exception as e:
                error_msg = str(e)
                # 如果是"not loaded"错误，先尝试释放再重试
                if "not loaded" in error_msg.lower() or "loading" in error_msg.lower():
                    try:
                        self.collection.release()
                        sleep(1)
                    except:
                        pass
                
                if attempt < max_retries - 1:
                    print(f"集合加载失败，重试中... ({attempt + 1}/{max_retries}): {e}")
                    sleep(wait_seconds)
                else:
                    raise Exception(f"集合加载最终失败: {e}")

    def _calculate_hash(self, content):
        """计算文档内容的MD5哈希值，用于去重判断"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    def _is_hash_exist(self, doc_hash):
        """检查文档哈希是否已存在于集合中"""
        try:
            res = self.collection.query(expr=f"doc_hash == '{doc_hash}'",output_fields=["id"])
            return len(res) > 0
        except Exception as e:
            print(f"检查哈希存在性出错: {e}")
            return False # 出错时默认不判定为重复，避免误判

    def add(self,chunks_emb_list):
        milvus_entities=[]
        try:
            for chunk in chunks_emb_list:
                chunk_text = chunk['text']
                doc_hash = self._calculate_hash(chunk_text) # 计算文档内容哈希
                # 检查该哈希是否已存在,内容去重
                if self._is_hash_exist(doc_hash):
                    #duplicate_count += 1
                    print(f"检测到重复文档切片 (id: {chunk['id']})，已跳过")
                else: # 添加到批量插入列表
                    milvus_entities.append({
                        "id": chunk['id'],  # 在得到一个节点是生成，可以直接在这里生成uuid  uuid.uuid4()
                        "vector": chunk['vector'],
                        "text": chunk_text,
                        "doc_id":chunk["doc_id"],
                        "doc_name":chunk["doc_name"],  # 文档名称，用于后续查询
                        "doc_hash": doc_hash # 存储哈希值
                    })
            # 批量插入向量到Milvus
            if milvus_entities:
                self.collection.insert(milvus_entities)
                self.collection.flush()
                print(f"已向Milvus插入 {len(milvus_entities)} 个新向量")
            else:
                print(f"已向Milvus插入 {len(milvus_entities)} 个新向量")
            return True
        except Exception as e:
            print(f"文档向量化报错:====={e}")
            raise

    def delete(self, task_id=None, use_id=None,task_node_id=None):
        """根据doc_id或vector_id删除数据或者chunk_id"""
        if not task_id and not use_id and not task_node_id:
            print("删除操作需要提供id")
            raise

        try:
            conditions = []
            if task_id: # 删除整个文档的方法，就提供doc id
                expr = f"task_id == '{task_id}'"
                conditions.append(expr)
                print(f"删除task_id为 {task_id} 的所有向量")
            if use_id: # 删除单个切片向量，能用chunk_id,因为是doc——id.父子命名法，不同文档父子那一块相同但是docid不同，所以不相同的chunk——id
                expr = f"use_id == '{use_id}'"
                conditions.append(expr)
                print(f"删除use_id为 {use_id} 的所有向量")

            exprs = " AND ".join(conditions) # 修改为联合删除，2025-10-30修改
            print(f"执行联合条件删除，过滤表达式：{exprs}")
            delete_res = self.collection.delete(exprs)
            self.collection.flush()
            delete_count=delete_res.delete_count if hasattr(delete_res, "delete_count") else len(delete_result)
            print(f"成功删除 {delete_count} 条数据")
            return True
        except Exception as e:
            print(f"删除操作出错: {e}")
            raise

    def update(self, task_node_id,chunk_emb,task_id=None,use_id=None): # old_vector_id
        """更新指定chunk的数据（主要用于更新文本内容和向量）"""
        # 输入时个字典格式
        try:# 先删除旧数据
            delete_success = self.delete(task_node_id=task_node_id,task_id=task_id,use_id=use_id) # vector_id=old_vector_id,
            if not delete_success:
                print(f"更新失败：无法删除旧数据 (task_node_id: {task_node_id})")
                raise
            # 插入新数据
            self.add([chunk_emb])  # 是一个字典结构数据
            print(f"成功更新task_node_id为 {task_node_id} 的向量数据")
            return True
        except Exception as e:
            print(f"更新操作出错: {e}")
            raise

    def close(self):# 释放Milvus集合
        if hasattr(self, 'collection'):
            self.collection.release()
            print("Milvus集合已释放")

# 混合检索方法
def hybrid_search(query_text, query_vector, collection_name='',uri='http://c-321c53c5d3badb26.milvus.aliyuncs.com:19530',top_k=10,filter_str='', timeout=120):
    '''
    混合检索函数，支持向量检索和BM25稀疏检索的融合
    参数:imeout: 集合加载超时时间（秒），None表示无限等待
    返回:list: 检索结果列表，每项包含 distance, task_node_id, task_id, use_id, doc_id, id, text
    返回示例:
        {
            'id': 'baefb998-f3ec-4858-81cf-a7f5dc3e9c67',
            'distance': 0.009900989942252636,
            'entity': {
                'doc_id': 'e93edd09-ff06-4c62-b2ba-ef9c22d34429',
                'text': ".算法评价 Dify上测试，数据稳定。",
                'chunk_id': 'e93edd09-ff06-4c62-b2ba-ef9c22d34429.2.15'
            }
        }
    '''
    client = MilvusClient(uri=uri,token=token) # "http://192.168.3.251:19530"
    # 先加载集合到内存（带重试和超时）
    max_retries = 3
    for attempt in range(max_retries):
        try:
            client.load_collection(collection_name=collection_name)
            # 等待加载完成
            #client.wait_for_loading_collection(collection_name=collection_name, timeout=None)
            print(f"集合 {collection_name} 加载成功")
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"集合加载失败: {e}")
            print(f"集合加载失败，重试中... ({attempt + 1}/{max_retries})")

    search_index_paras={"metric_type": "IP"}
    bm25_index_paras={"metric_type": "BM25"}
    if filter_str.strip():
        res1 = AnnSearchRequest([query_vector], "vector", search_index_paras, limit=top_k,expr=filter_str.strip())  #条件删选 "doc_id == $doc_id"
        res2 = AnnSearchRequest([query_text], "sparse_bm25", bm25_index_paras, limit=top_k,expr=filter_str.strip())
    else:#全局检索 filter_str='doc_id="%s"' :'doc_id == "e93edd09-ff06-4c62-b2ba-ef9c22d34429"' or 'doc_id in ["doc_id_1", "doc_id_2", "doc_id_3"]'
        res1 = AnnSearchRequest([query_vector], "vector", search_index_paras, limit=top_k)  #条件删选
        res2 = AnnSearchRequest([query_text], "sparse_bm25", bm25_index_paras, limit=top_k)

    reqs = [res1, res2]
    ranker = RRFRanker(100)
    hybrid_search_res = client.hybrid_search(
        collection_name=collection_name,
        reqs=reqs,
        ranker=ranker,
        limit=top_k,
        output_fields=["text","id","doc_name"]
    )
    res_out=[]
    for hits in hybrid_search_res:
        print(type(hits),len(hits))
        for hit in hits:
            #print(hit,type(hit))
            chunk_info = {
                "distance": hit.distance, #hit.distance, #  # 可选：保留相似度分数
                "id": hit.entity.id,
                "text": hit.entity.text,
                "doc_name": hit.entity.doc_name,  # 文档名称，用于后续查询
            }
            res_out.append(chunk_info)
    print("检索结果数量：=====",len(res_out))
    client.close()
    return res_out
    
def insert_batch(chunk_emb_list,collection_name,alias_name="default",env=''):
    '''
    向量化的数据维度,     "doc_hash": doc_hash # 存储哈希值,再add过程计算
    {
        "id": chunk['id'],
        "vector": chunk['vector'],
        "text": chunk_text,

   
    }
    '''
    mvc = myMilvus(collection_name)
    mvc.add(chunk_emb_list)
    print("入库完成！")
    mvc.close()
    return 

if __name__=="__main__":
    print("ok")
    import uuid
    from utils import get_embed
    '''emb_res=get_embed('http://192.168.3.251:8020/embed','墙体排风设计说明') # '水电设计详情说明'
    if emb_res['success']:
        embeding=emb_res['data']['embeddings']  # 1维数组
        testdata=[
            {
                "id": str(uuid.uuid4()),
                "vector": embeding,
                "text": '墙体排风设计说明',
                "task_id":1,
                "use_id": str(uuid.uuid4()),
                "task_node_id": 2
            }
        ]
        insert_batch(testdata,collection_name='testmilvus',alias_name="default",env='')
    else:
        print("有错误啊！",emb_res)'''
    query="墙体排风"
    query_emb=get_embed('http://192.168.3.251:8020/embed', query)
    if query_emb['success']:
        embeding=query_emb['data']['embeddings']  # 1维数组
        out=hybrid_search(query, embeding, collection_name='testmilvus',uri='http://192.168.3.251:19530',top_k=10,filter_str='')
    print("result:",out)
    print("over!!!!!!!!!!")