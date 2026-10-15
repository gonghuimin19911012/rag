import asyncio
import json
import re
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 自己的代码
from config import llm_url, llm_model, collection_name, bge_model, api_key, MYSQL_CONFIG
from embedmodel import BGEEmbedder, get_embed
from milvus_main import hybrid_search
from sql_main import kgSQL, insert_conversation, insert_message

# ===================== 全局配置 =====================
type_label=1  # 1:问数，2:专业知识问答，3：图纸问答
LLM_MODEL = llm_model
LLM_BASE_URL = llm_url
LLM_API_KEY = api_key
MODEL_PATH = bge_model["path"]

embedder = BGEEmbedder(MODEL_PATH)
mysql = kgSQL(database=MYSQL_CONFIG["database"])

# key: f"{user_id}:{conversation_id}:{project_id}"
USER_HISTORY_CACHE: Dict[str, List[Dict[str, Any]]] = {}
HISTORY_LOCK = asyncio.Lock()
HISTORY_TTL_SECONDS = 24 * 3600
MAX_HISTORY_ROUNDS = 10

# ===================== FastAPI =====================
app = FastAPI(title="非流式 Text2SQL 生成服务")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================== 请求模型 =====================
class QueryRequest(BaseModel):
    user_question: str = Field(..., min_length=1, description="用户自然语言问题")
    project_id: int = Field(..., description="项目ID，用于生成SQL项目隔离条件")
    user_id: str = Field(..., min_length=1, description="用户ID")
    conversation_id: str = Field(..., min_length=1, description="会话ID")
    top_k: int = Field(10, description="返回的上下文数量")


# ===================== 返回工具 =====================
def make_response(
    *,
    message_id: str,
    text: str = "",
    answer: str = "",
    sources: List[Dict[str, Any]] = [],
    error: str = "",
) -> Dict[str, Any]:
    return {
        "type": "done" if not error else "error",
        "text": text,
        "message_id": message_id,
        "answer": answer,
        "sources": sources,
        "error": error,
    }


# ===================== 历史管理 =====================
def build_history_key(user_id: str, conversation_id: str, project_id: int) -> str:
    return f"kg:{user_id}:{conversation_id}:{project_id}"


async def get_history_text(history_key: str, limit_rounds: int = MAX_HISTORY_ROUNDS) -> str:
    async with HISTORY_LOCK:
        items = USER_HISTORY_CACHE.get(history_key, [])[-limit_rounds * 2:]

    if not items:
        return ""

    lines: List[str] = []
    for item in items:
        role = "User" if item.get("role") == "user" else "AI"
        content = item.get("content", "")
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def append_history(history_key: str, role: str, content: str) -> None:
    now = time.time()
    async with HISTORY_LOCK:
        USER_HISTORY_CACHE.setdefault(history_key, []).append({
            "role": role,
            "content": content,
            "create_time": now,
        })
        USER_HISTORY_CACHE[history_key] = USER_HISTORY_CACHE[history_key][-MAX_HISTORY_ROUNDS * 2:]


async def clean_expired_history_once() -> None:
    now = time.time()
    async with HISTORY_LOCK:
        expired_keys = [
            key for key, items in USER_HISTORY_CACHE.items()
            if not items or now - items[-1].get("create_time", 0) > HISTORY_TTL_SECONDS
        ]
        for key in expired_keys:
            USER_HISTORY_CACHE.pop(key, None)


async def history_cleaner_loop() -> None:
    while True:
        await asyncio.sleep(3600)
        await clean_expired_history_once()


@app.on_event("startup")
async def startup_event() -> None:
    asyncio.create_task(history_cleaner_loop())


# ===================== 向量化与检索 =====================
def embedding_function(data: str):
    vecres = get_embed(data, embedder)
    if vecres and vecres.get("success"):
        return vecres.get("embeddings")
    return None


async def retrieve_ddl_node(user_question: str, query_vector, collect_name: str, project_id: int) -> str:
    """
    全库检索知识库，不加项目过滤
    """
    try:
        filter_str = ""
        retrieved_ddl = await asyncio.to_thread(
            hybrid_search,
            query_text=user_question,
            query_vector=query_vector,
            collection_name=collect_name,
            filter_str=filter_str,
            top_k=10,
        )
        
        return retrieved_ddl or ""
    except Exception as e:
        print(f"项目{project_id} 专业知识检索失败:", e)
        return ""

# ===================== LLM 知识问答 生成：非流式 =====================
async def generate_answer(llm: ChatOpenAI,query: str,context: str,project_id: int,history: str = "",) -> str:
    if context:
        sql_prompt = PromptTemplate.from_template(
            """你是一个专业的建筑工程设计造价师，帮助用户回答关于建筑工程计量计价的问题。
            #相关文档：{context}
            #历史对话(可为空):{history}
            #当前时间：{current_time}
            #用户问题：{query}
            请根据提供的相关文档和历史对话,输出有帮助的回答。
            如果已提供的数据中不包含相关信息，请根据你的知识经验回答问题，但要强调"根据提供的文档，我无法找到相关信息。以下是我的回答经验："，说明这一点。
            回答：
            """
        )
    else:
        sql_prompt = PromptTemplate.from_template(
            """你是一个专业的建筑工程设计造价师，帮助用户回答关于建筑工程计量计价的问题。
            #历史对话(可为空)：{history_text}
            #当前时间：{current_time}
            #用户问题：{query}
            请根据提供的历史对话输出有帮助的回答，禁止瞎编。
            回答：
            """
        )
    chain = sql_prompt | llm | StrOutputParser()
    result = await chain.ainvoke({
        "context": context,
        "project_id": project_id,
        "history": history,
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
    })
    print("生成的回答:", len(result))
    return result


# ===================== API：非流式 =====================
@app.post("/qa_ask")
async def qaAsk(request: QueryRequest):
    message_id = str(uuid.uuid4())
    user_question = request.user_question.strip()
    conversation_id = request.conversation_id or str(uuid.uuid4())
    user_id = request.user_id
    project_id = request.project_id
    history_key = build_history_key(user_id, conversation_id, project_id)
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        # 只记录会话,会话表
        conversation_info = {
            "id": conversation_id,
            "create_time": start_time,
            "use_id": user_id,
            "task_id": project_id,
            "info": user_question,
            "session_id": conversation_id,
            "type": type_label,
        }
        try:
            insert_conversation(mysql, conversation_info)
        except Exception as e:
            # conversation 重复插入等问题不应该影响 SQL 生成。
            print("insert_conversation 失败或已存在:", e)# 存在就不插入
            #return make_response(message_id=message_id, error=f"会话记录失败:{e}")

        query_vector = await asyncio.to_thread(embedding_function, user_question)
        if query_vector is None:
            return make_response(message_id=message_id, error="向量化失败")

        history_text = await get_history_text(history_key)
        contexts = await retrieve_ddl_node(
            user_question=user_question,
            query_vector=query_vector,
            collect_name=collection_name,
            project_id=project_id,
        )
        sources = [] # 对检索的结果进行溯源
        context_list = []
        if not contexts:
            print("未检索到相关信息")
            #return make_response(message_id=message_id,text="未检索到相关信息",)
        else:# 构建上下文
            for i, result in enumerate(contexts[:request.top_k]):
                text = result.get('text', '')
                score=result.get('distance', 0.0)
                if text.strip():
                    context_list.append(f"[来源 {i+1}：相关度：{score}]: {text}")
                sources.append({
                    "id": i + 1,
                    "text": text[:200] + "..." if len(text) > 200 else text,
                    "score": score,
                    "doc_id": result.get('doc_id', ''),
                })

        llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            temperature=0.3,
            streaming=False,
            timeout=300,
            max_retries=1,
        )

        final_ans = await generate_answer(
            llm=llm,
            query=user_question,
            context=context_list, #contexts
            project_id=project_id,
            history=history_text,
        )
        print("llm生成结果：",final_ans)

        await append_history(history_key, "user", user_question)
        await append_history(history_key, "assistant", final_ans)
        message_info={
            "id": message_id,
            "doc_id": conversation_id,
            "create_time": start_time,
            "info": {"question": user_question,"answer": final_ans},
            "task_id": project_id,
            "type": type_label,
        }
        # 可选：记录 message。如果你的 insert_message 参数结构不同，可以按你的函数定义调整。
        try:
            insert_message(mysql, message_info)
        except Exception as e:
            print("insert_message 失败:", e)

        return make_response(
            message_id=message_id,
            text="专业知识问答",
            answer=final_ans,
            sources=sources,
        )
    except Exception as e:
        return make_response(message_id=message_id, error=f"SQL生成异常：{str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9080)