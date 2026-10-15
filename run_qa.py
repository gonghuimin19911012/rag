
"""
独立RAG问答服务
文件: rag_qa_service.py
功能: 提供基于知识库的问答服务
"""
import os
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Query, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn


# 导入自定义模块
from milvus_main import hybrid_search
from openai import OpenAI
from sql_main import kgSQL,insert_conversation,insert_message
from embedmodel import BGEEmbedder,get_embed
from config import llm_url,llm_model,api_key,collection_name,bge_model,MYSQL_CONFIG

Model_path=bge_model['path']  # 向量模型路径
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
#初始化数据库和向量模型
db = kgSQL(database=MYSQL_CONFIG['database'])
embedder = BGEEmbedder(Model_path)
type_label=2  # 1:问数，2:专业知识问答，3：图纸问答
# ====================== 配置 ======================
class Config:
    """配置类"""
    LLM_URL = llm_url
    API_KEY = api_key
    LLM_MODEL = llm_model
    MILVUS_COLLECTION=collection_name

config = Config()

# ====================== 数据模型 ======================
class QuestionRequest(BaseModel):
    """问答请求模型"""
    task_id: int = Field(..., description="任务ID")
    use_id: str = Field(..., description="用户ID")
    doc_id: str = Field(None, description="文档ID")
    question: str = Field(..., description="问题内容")
    top_k: int = 10 # Field(5, description="检索结果数量", ge=1, le=20)
    temperature: float = 0.5 #  Field(0.1, description="温度参数", ge=0.0, le=1.0)
    max_tokens: int = 10240       #Field(1000, description="最大token数", ge=100, le=4000)
    
class ChatRequest(BaseModel):
    """对话请求模型"""
    session_id: str = Field(..., description="会话ID")
    message: str = Field(..., description="用户消息")
    task_id: Optional[int] = Field(None, description="任务ID（可选）")
    use_id: Optional[str] = Field("anonymous", description="用户ID")
    top_k: int =5 # Field(5, description="检索结果数量")
    temperature: float =0.7 # Field(0.1, description="温度参数")
    max_tokens: int = 10240 #Field(1000, description="最大token数")

class AnswerResponse(BaseModel):
    """回答响应模型"""
    success: bool = Field(..., description="是否成功")
    answer: str = Field(..., description="回答内容")
    sources: List[Dict[str, Any]] = Field([], description="来源文档")
    question: str = Field(..., description="原始问题")
    session_id: Optional[str] = Field(None, description="会话ID")
    timestamp: str = Field(..., description="时间戳")
    processing_time: float = Field(..., description="处理时间(秒)")
    error: Optional[str] = Field(None, description="错误信息")

class ChatResponse(BaseModel):
    """对话响应模型"""
    success: bool = Field(..., description="是否成功")
    session_id: str = Field(..., description="会话ID")
    answer: str = Field(..., description="回答内容")
    message_id: str = Field(..., description="消息ID")
    timestamp: str = Field(..., description="时间戳")
    processing_time: float = Field(..., description="处理时间(秒)")

# ====================== 服务类 ======================

class RAGQAService:
    """RAG问答服务类"""
    def __init__(self,):
        # 初始化LLM客户端
        self.llm_client = OpenAI(api_key=config.API_KEY,base_url=config.LLM_URL)
        # 会话历史存储
        self.sessions = {}
        logger.info("RAG问答服务初始化完成")
    
    def get_answer(self, request: QuestionRequest) -> AnswerResponse:
        """获取问题答案"""
        start_time = datetime.now()
        try:
            # 1. 向量化问题
            query_vector = get_embed(request.question,embedder)
            
            if not query_vector.get('success'):
                error_msg = f"向量化失败: {query_vector.get('error', '未知错误')}"
                logger.error(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)
            
            # 2. 在Milvus中检索,知识问答就是全库检索，没有固定的文档id，除非固定选择
            #filter_str = f"doc_id == '{request.doc_id}'" if request.doc_id else ''
            #filter_str = f'task_id == {request.task_id}' if request.task_id else ''
            filter_str =''

            search_results = hybrid_search(
                query_text=request.question,
                query_vector=query_vector['embeddings'],
                collection_name=config.MILVUS_COLLECTION,
                filter_str=filter_str,
                top_k=request.top_k
            )   # list_dict
            
            if not search_results:
                logger.warning(f"未找到相关文档: doc_id={request.doc_id}")
                # 如果没有检索到相关文档，使用空上下文
                context = ""
                sources = []
            else:# 构建上下文
                contexts = []
                sources = []
                for i, result in enumerate(search_results[:request.top_k]):
                    text = result.get('text', '')
                    if text:
                        contexts.append(f"[来源 {i+1}]: {text}")
                    sources.append({
                        "id": i + 1,
                        "text": text[:200] + "..." if len(text) > 200 else text,
                        "score": result.get('distance', 0.0),
                        "doc_id": result.get('doc_id', ''),
                    })
                
                context = "\n\n".join(contexts)
                logger.info(f"检索到 {len(sources)} 个相关文档片段")
            
            # 3. 构建prompt
            prompt = self._build_prompt(request.question, context)
            
            # 4. 调用LLM生成答案
            answer = self._call_llm(
                prompt=prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            answer=answer.encode("utf-8", "ignore").decode("utf-8") # 去掉emoji表情符号
            # 5. 记录问答历史
            self._record_qa_history(request, answer, sources)
            
            # 计算处理时间
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"问题回答完成: 问题='{request.question[:50]}...', 用时={processing_time:.2f}s")
            #insert_message
            return AnswerResponse(
                success=True,
                answer=answer,
                sources=sources,
                question=request.question,
                timestamp=datetime.now().isoformat(),
                processing_time=processing_time
            )
            
        except HTTPException:
            raise
        except Exception as e:
            error_msg = f"ask 问答处理失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            processing_time = (datetime.now() - start_time).total_seconds()
            return AnswerResponse(
                success=False,
                answer="抱歉，我遇到了一个内部错误，无法回答您的问题。",
                sources=[],
                question=request.question,
                timestamp=datetime.now().isoformat(),
                processing_time=processing_time,
                error=error_msg
            )
    
    def chat(self, request: ChatRequest) -> ChatResponse:
        """多轮对话"""
        start_time = datetime.now()
        message_id = str(uuid.uuid4())
        try:
            # 1. 初始化或获取会话历史
            if request.session_id not in self.sessions:
                self.sessions[request.session_id] = []
            
            session_history = self.sessions[request.session_id]
            
            # 2. 向量化问题
            query_vector = get_embed(request.message,embedder)
            if not query_vector.get('success'):
                error_msg = f"向量化失败: {query_vector.get('error', '未知错误')}"
                logger.error(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)
            
            # 3. 在Milvus中检索
            # 如果没有指定任务，可以放宽检索条件
            filter_str = f'task_id == {request.task_id}' if request.task_id else ''
            
            search_results = hybrid_search(
                query_text=request.message,
                query_vector=query_vector['embeddings'],
                collection_name=config.MILVUS_COLLECTION,
                filter_str=filter_str,
                top_k=request.top_k
            )
            # 4. 构建上下文
            context = ""
            if search_results:
                contexts = []
                for i, result in enumerate(search_results[:request.top_k]):
                    text = result.get('text', '')
                    if text:
                        contexts.append(f"[来源 {i+1}]: {text}")
                context = "\n\n".join(contexts)
            
            # 5. 构建带历史记录的prompt
            prompt = self._build_chat_prompt(
                message=request.message,
                context=context,
                history=session_history[-5:]  # 最近5轮对话
            )
            # 6. 调用LLM生成答案
            answer = self._call_llm(
                prompt=prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            answer=answer.encode("utf-8", "ignore").decode("utf-8") # 去掉emoji表情符号
            # 7. 更新会话历史
            session_history.append({
                "role": "user",
                "message": request.message,
                "timestamp": datetime.now().isoformat()
            })
            session_history.append({
                "role": "assistant",
                "message": answer,
                "timestamp": datetime.now().isoformat()
            })
            # 限制历史记录长度
            if len(session_history) > 20:  # 最多保存10轮对话
                self.sessions[request.session_id] = session_history[-20:]
            # 8. 记录聊天历史
            self._record_chat_history(request, answer, message_id)
            # 计算处理时间
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"对话完成: session_id={request.session_id}, 消息ID={message_id}, 用时={processing_time:.2f}s")
            
            return ChatResponse(
                success=True,
                session_id=request.session_id,
                answer=answer,
                message_id=message_id,
                timestamp=datetime.now().isoformat(),
                processing_time=processing_time
            )
        except HTTPException:
            raise
        except Exception as e:
            error_msg = f"chat 对话处理失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise HTTPException(status_code=500, detail=error_msg)
    
    def _build_prompt(self, question: str, context: str) -> str:
        """构建prompt"""
        if context:
            prompt = f"""你是一个专业的建筑工程造价师，请基于以下提供的内容回答问题。如果不包含相关信息，请根据你的知识回答，但要说明这一点。
            #相关文档：{context}
            #问题：{question}
            请提供。如果相关文档没有相关信息，请说"根据提供的文档，我无法找到相关信息。"
            回答：
            """
        else:
            prompt = f"""请根据问题输出结果：
            #问题：{question}
            请根据你的知识经验提供准确、有用的回答。
            回答：
            """
        return prompt
    
    def _build_chat_prompt(self, message: str, context: str, history: List[Dict]) -> str:
        """构建对话prompt"""
        # 构建历史对话
        history_text = ""
        for item in history:
            role = "用户" if item["role"] == "user" else "助手"
            history_text += f"{role}: {item['message']}\n\n"
        if context:
            prompt = f"""你是一个专业的建筑工程设计造价师，帮助用户回答关于建筑工程计量计价的问题。
            #相关文档：{context}
            #历史对话：{history_text}
            #用户问题：{message}
            请根据提供的相关文档和历史对话提供有帮助的回答。
            如果已提供的数据中不包含相关信息，请根据你的知识经验回答问题，但要强调"根据提供的文档，我无法找到相关信息。以下是我的回答经验："，说明这一点。
            回答：
            """
        else:
            prompt = f"""你是一个专业的AI助手，帮助用户回答关于建筑工程计量计价的问题。
            #历史对话：{history_text}
            #用户问题：{message}
            请根据提供的历史对话提供有帮助的回答。
            助手：
            """
        return prompt
    
    def _call_llm(self, prompt: str, temperature: float = 0.7, max_tokens: int = 10000) -> str:
        """调用LLM生成回答"""
        try:
            completion = self.llm_client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            answer = completion.choices[0].message.content.strip()
            return answer
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            raise HTTPException(status_code=500, detail=f"LLM调用失败: {str(e)}")
    
    def _record_qa_history(self, request: QuestionRequest, answer: str, sources: List[Dict]):
        """记录问答历史到数据库"""
        try:
            qa_data = {
                "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "id": f'{request.use_id}_{str(uuid.uuid4())}', # "use_id
                "task_id": request.task_id,
                "doc_id": request.doc_id,  # doc_id"question": request.question,"answer"
                "info": json.dumps({"q":request.question,"a":answer,'sources':sources},ensure_ascii=False),
                "type": type_label,
            }
            # 假设数据库中有qa_history表
            insert_message(db,qa_data) #id, create_time, conversation_id/doc_id, task_id, info
            logger.debug(f"问答历史记录: {qa_data}")
        except Exception as e:
            logger.error(f"记录问答历史失败: {e}")
            # 不抛出异常，避免影响主流程
    
    def _record_chat_history(self, request: ChatRequest, answer: str, message_id: str):
        """记录聊天历史到数据库"""
        try:
            chat_data = {
                "id": message_id,
                "session_id": request.session_id, # 一个对话框有多个message
                "use_id": request.use_id,
                "task_id": request.task_id,
                "info": json.dumps({"q":request.message,"a":answer},ensure_ascii=False), #  "message": request.message,"answer": answer,
                "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": type_label,
            }
            
            # 假设数据库中有chat_history表 (id, create_time, use_id, task_id, info，session_id)
            insert_conversation(db,chat_data)
            logger.debug(f"聊天历史记录: {chat_data}")
            
        except Exception as e:
            logger.error(f"记录聊天历史失败: {e}")
            # 不抛出异常，避免影响主流程
    
    def clear_session(self, session_id: str) -> bool:
        """清除指定会话的历史"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"已清除会话: {session_id}")
            return True
        return False
    
    def get_session_info(self, session_id: str) -> Dict:
        """获取会话信息"""
        if session_id in self.sessions:
            history = self.sessions[session_id]
            return {
                "session_id": session_id,
                "history_count": len(history),
                "last_update": history[-1]["timestamp"] if history else None
            }
        return {"session_id": session_id, "history_count": 0, "last_update": None}

# ====================== FastAPI应用 ======================
app = FastAPI(
    title="RAG问答服务",
    description="基于知识库的智能问答服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化服务
rag_service = RAGQAService()

# ====================== 路由 ======================
@app.get("/")
async def root():
    """根路径，返回服务信息"""
    return {
        "service": "RAG问答服务",
        "version": "1.0.0",
        "status": "运行中",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "问答接口": "/api/ask",
            "对话接口": "/api/chat",
            "文档搜索接口": "/api/search",
            "清除会话": "/api/chat/clear/{session_id}",
            "会话信息": "/api/chat/info/{session_id}",
            "健康检查": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "RAG问答服务"
    }

@app.post("/api/qa/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    单次问答接口
    基于知识库回答用户问题
    """
    logger.info(f"收到问答请求: task_id={request.task_id}, doc_id={request.doc_id}")
    logger.debug(f"问题: {request.question}")
    
    response = rag_service.get_answer(request)
    
    return response

@app.post("/api/qa/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    对话接口
    支持多轮对话的问答
    """
    logger.info(f"收到对话请求: session_id={request.session_id} task_id={request.task_id}")
    logger.debug(f"消息: {request.message}")
    
    response = rag_service.chat(request)
    return response

@app.get("/api/chat/clear/{session_id}")
async def clear_chat_session(session_id: str):
    """
    清除指定会话的历史记录
    """
    success = rag_service.clear_session(session_id)
    
    return {
        "success": success,
        "session_id": session_id,
        "message": "会话已清除" if success else "会话不存在"
    }

@app.get("/api/chat/info/{session_id}")
async def get_chat_session_info(session_id: str):
    """
    获取会话信息
    """
    info = rag_service.get_session_info(session_id)
    
    return {
        "success": True,
        "session_info": info
    }

@app.get("/api/qa/search")
async def search_documents(
    query: str = Query(..., description="搜索查询"),
    task_id: Optional[int] = Query(None, description="任务ID"),
    doc_id: Optional[str] = Query(None, description="文档ID"),
    top_k: int = Query(5, description="返回结果数量")
):
    """
    文档搜索接口
    直接在知识库中搜索相关文档片段
    """
    try:
        # 向量化查询
        query_vector = get_embed(query,embedder)
        if not query_vector.get('success'):
            raise HTTPException(status_code=500, detail="向量化失败")
        
        # 构建过滤条件
        filter_parts = []
        if task_id:
            filter_parts.append(f'task_id == {task_id}')
        if doc_id:
            filter_parts.append(f'doc_id == "{doc_id}"')

        filter_str = " && ".join(filter_parts) if filter_parts else ''
        
        # 执行搜索
        search_results = hybrid_search(
            query_text=query,
            query_vector=query_vector['embeddings'],
            collection_name=config.MILVUS_COLLECTION,
            filter_str=filter_str,
            top_k=top_k
        )
        
        # 格式化结果
        formatted_results = []
        for i, result in enumerate(search_results):
            formatted_results.append({
                "id": i + 1,
                "text": result.get('text', ''),
                "score": result.get('distance', 0.0),
                "doc_id": result.get('doc_id', ''), #"task_id": result.get('task_id', '')
            })
        
        return {
            "success": True,
            "query": query,
            "results": formatted_results,
            "count": len(formatted_results)
        }
        
    except Exception as e:
        logger.error(f"搜索失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

# ====================== 启动脚本 ======================
if __name__ == "__main__":
    logger.info("启动RAG专业知识问答服务...")
    logger.info(f"服务地址: http://0.0.0.0:9080")
    logger.info(f"API文档: http://0.0.0.0:9080/docs")
    
    uvicorn.run(
        "run_qa:app",
        host='0.0.0.0',
        port=9080,
        workers=2,
        log_level="info"
    )