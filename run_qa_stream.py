import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional
import requests

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 自己的代码
from config import llm_url, llm_model, collection_name, bge_model, api_key,embed_url, MYSQL_CONFIG
from milvus_main import hybrid_search
from sql_main import kgSQL, insert_conversation, insert_message

# ===================== 全局配置 =====================
TYPE_LABEL = 2  # 1:问数，2:专业知识问答，3：图纸问答
LLM_MODEL = llm_model
LLM_BASE_URL = llm_url
LLM_API_KEY = api_key
#mysql = kgSQL(database=MYSQL_CONFIG["database"])
# key: f"kg:{user_id}:{conversation_id}:{project_id}"
USER_HISTORY_CACHE: Dict[str, List[Dict[str, Any]]] = {}
HISTORY_LOCK = asyncio.Lock()
HISTORY_TTL_SECONDS = 24 * 3600
MAX_HISTORY_ROUNDS = 10

# ===================== FastAPI =====================
app = FastAPI(title="流式 SSE 专业知识问答服务")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 不要全局mysql，每次新建连接
def _insert_conv_safe(func,info):
    db = kgSQL(database=MYSQL_CONFIG["database"])
    try:
        return func(db, info)
    finally:
        db.close()  # 强制关闭连接，释放资源
# ===================== 请求模型 =====================
class QueryRequest(BaseModel):
    user_question: str = Field(..., min_length=1, description="用户自然语言问题")
    project_id: int = Field(..., description="项目ID")
    user_id: str = Field(..., min_length=1, description="用户ID")
    conversation_id: Optional[str] = Field(None, description="会话ID；为空则自动生成")
    top_k: int = Field(10, ge=1, le=50, description="返回的上下文数量")


# ===================== SSE 返回工具 =====================
def sse_data(payload: Dict[str, Any]) -> Dict[str, str]:
    """EventSourceResponse 推荐 yield dict，data 放 JSON 字符串。"""
    return {
        "event": "message",
        "data": json.dumps(payload, ensure_ascii=False),
    }


def make_done(
    *,
    message_id: str,
    answer: str = "",
    sources: Optional[List[Dict[str, Any]]] = None,
    text: str = "专业知识问答",
) -> Dict[str, Any]:
    return {
        "type": "done",
        "text": text,
        "message_id": message_id,
        "answer": answer,
        "sources": sources or [],
        "error": "",
    }


async def error_generator(error: str, message_id: str) -> AsyncGenerator[Dict[str, str], None]:
    yield sse_data({
        "type": "error",
        "text": error,
        "message_id": message_id,
        "answer": "",
        "sources": [],
        "error": error,
    })
    yield sse_data(make_done(message_id=message_id, answer="", sources=[], text=""))


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
'''def embedding_function(data: str):
    vecres = get_embed(data, embedder)
    if vecres and vecres.get("success"):
        return vecres.get("embeddings")
    return None'''
def embedding_function(data: str): #embed_url
    start_time = time.time()
    headers = {"Content-Type": "application/json",}
    payload={"texts":data}
    response = requests.post(url=embed_url, headers=headers,json=payload,timeout=600)
    response_time = time.time() - start_time
    #logger.debug
    print(f"请求 {embed_url} 完成，耗时: {response_time:.3f}秒，状态码: {response.status_code}")
    if response.status_code == 200:
        res_data = response.json()# 直接返回向量列表
        print("embedding_function success:",len(res_data.get("embeddings")))
        return res_data.get("embeddings")
    else:
        error_msg = f"请求失败，状态码: {response.status_code}"
        print("embedding_function error:",error_msg)
        return None #{"success": False,"data":error_msg}

async def retrieve_knowledge_node(
    user_question: str,
    query_vector,
    collect_name: str,
    project_id: int,
    top_k: int,
):
    """全库检索知识库，不加项目过滤。"""
    try:
        retrieved = await asyncio.to_thread(
            hybrid_search,
            query_text=user_question,
            query_vector=query_vector,
            collection_name=collect_name,
            filter_str="",
            top_k=top_k,
        )
        return retrieved or []
    except Exception as e:
        print(f"项目{project_id} 专业知识检索失败:", e)
        return []


def build_context_and_sources(contexts: Any, top_k: int):
    sources: List[Dict[str, Any]] = []
    context_list: List[str] = []

    if not contexts:
        return "", sources

    # hybrid_search 可能返回 list[dict]，也可能返回字符串，这里做兼容。
    if isinstance(contexts, str):
        return contexts, [{"id": 1, "text": contexts[:200] + "..." if len(contexts) > 200 else contexts, "score": 0.0, "doc_id": ""}]

    for i, result in enumerate(contexts[:top_k]):
        if isinstance(result, dict):
            text = result.get("text", "") or result.get("content", "") or ""
            score = result.get("distance", result.get("score", 0.0))
            doc_name = result.get("doc_name", "")
        else:
            text = str(result)
            score = 0.0
            doc_name = ""

        if text.strip():
            context_list.append(f"[来源 {i + 1}，相关度：{score}]\n{text}")
            sources.append({
                "id": i + 1,
                "text": text[:150] + "..." if len(text) > 150 else text,
                "score": score,
                "doc_name": doc_name, # 文档名称，用于后续查询
            })

    return "\n\n".join(context_list), sources


# ===================== LLM 知识问答：流式 =====================
async def generate_answer_stream(llm: ChatOpenAI,query: str,context: str,project_id: int,history: str = "",) -> AsyncGenerator[str, None]:
    if context:
        prompt = PromptTemplate.from_template(
            """
            你是一个专业的建筑工程设计造价师，帮助用户回答关于建筑工程计量计价的问题。
            【相关文档】{context}
            【历史对话】{history}
            【当前时间】{current_time}
            【用户问题】{query}
            请根据提供的相关文档和历史对话，输出有帮助的回答。
            如果提供的文档中没有相关信息，可以根据你的专业经验回答，但必须先说明："根据提供的文档，我无法找到相关信息。以下是我的经验性回答："
            回答：
            """
        )
    else:
        prompt = PromptTemplate.from_template(
            """
            你是一个专业的建筑工程设计造价师，帮助用户回答关于建筑工程计量计价的问题。
            【历史对话】{history}
            【当前时间】{current_time}
            【用户问题】{query}
            当前没有检索到相关文档。请基于历史对话和专业知识回答，但不要编造具体规范条文、数据来源或不存在的项目资料。
            回答：
            """
        )

    chain = prompt | llm | StrOutputParser()
    async for token in chain.astream({
        "context": context,
        "project_id": project_id,
        "history": history,
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
    }):
        # 有些 OpenAI-compatible 服务会推空 token，过滤掉。
        if token is None or token == "":
            continue
        yield token


# ===================== API：SSE 流式 =====================
@app.post("/qa_ask")
async def qaAsk(request: QueryRequest):
    message_id = str(uuid.uuid4())
    user_question = request.user_question.strip()
    conversation_id = request.conversation_id or str(uuid.uuid4())
    user_id = request.user_id
    project_id = request.project_id
    history_key = build_history_key(user_id, conversation_id, project_id)
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    async def stream() -> AsyncGenerator[Dict[str, str], None]:
        final_answer = ""
        sources: List[Dict[str, Any]] = []

        try:
            yield sse_data({"type": "start","text": "开始处理问题","message_id": message_id,"answer": "","sources": [],"error": "",})

            # 记录会话：重复插入不影响问答。
            conversation_info = {
                "id": conversation_id,
                "create_time": start_time,
                "use_id": user_id,
                "task_id": project_id,
                "info": user_question,
                "session_id": conversation_id,
                "type": TYPE_LABEL,
            }
            try:
                await asyncio.to_thread(_insert_conv_safe, insert_conversation, conversation_info)#await asyncio.to_thread(insert_conversation, mysql, conversation_info)
            except Exception as e:
                print("insert_conversation 失败或已存在:", e)

            #yield sse_data({"type": "progress","text": "正在向量化问题","message_id": message_id,"answer": "","sources": [],"error": "",})

            query_vector = await asyncio.to_thread(embedding_function, user_question)
            if query_vector is None:
                yield sse_data({"type": "error","text": "向量化失败","message_id": message_id,"answer": "","sources": [],"error": "向量化失败",})
                yield sse_data(make_done(message_id=message_id, answer="", sources=[]))
                return

            history_text = await get_history_text(history_key)
            #yield sse_data({"type": "progress","text": "正在检索相关文档","message_id": message_id,"answer": "","sources": [],"error": "",})
            contexts = await retrieve_knowledge_node(
                user_question=user_question,
                query_vector=query_vector,
                collect_name=collection_name,
                project_id=project_id,
                top_k=request.top_k,
            )
            context_text, sources = build_context_and_sources(contexts, request.top_k)
            yield sse_data({"type": "sources","text": "检索完成","message_id": message_id,"answer": "","sources": sources,"error": "",})
            llm = ChatOpenAI(
                model=LLM_MODEL,
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL,
                temperature=0.3,
                streaming=True,
                timeout=120,
                max_retries=1,
            )
            #yield sse_data({"type": "progress","text": "正在生成回答","message_id": message_id,"answer": "","sources": [],"error": "",})
            async for token in generate_answer_stream(llm=llm,query=user_question,context=context_text,project_id=project_id,history=history_text,):
                final_answer += token
                yield sse_data({
                    "type": "answer",
                    "text": token,
                    "message_id": message_id,
                    "answer": token,
                    "sources": [],
                    "error": "",
                })

            if not final_answer.strip():
                final_answer = "未生成有效回答。"

            await append_history(history_key, "user", user_question)
            await append_history(history_key, "assistant", final_answer)
            print(f"问题: {user_question}, 回答: {final_answer}")
            message_info = {
                "id": message_id,
                "doc_id": conversation_id,
                "create_time": start_time,
                # 如果你的数据库字段是 JSON 类型，也可以直接传 dict；TEXT 字段建议传字符串。
                "info": json.dumps({"question": user_question, "answer": final_answer}, ensure_ascii=False),
                "task_id": project_id,
                "type": TYPE_LABEL,
            }
            try:
                await asyncio.to_thread(_insert_conv_safe, insert_message, message_info) #await asyncio.to_thread(insert_message, mysql, message_info)
            except Exception as e:
                print("insert_message 失败:", e)

            yield sse_data(make_done(message_id=message_id,answer=final_answer,sources=sources,text="专业知识问答",))

        except Exception as e:
            err = f"问答生成异常：{str(e)}"
            print(err)
            yield sse_data({"type": "error","text": err,"message_id": message_id,"answer": "","sources": sources,"error": err,})
            yield sse_data(make_done(message_id=message_id, answer=final_answer, sources=sources, text=""))

    return EventSourceResponse(stream())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("run_qa_stream:app", host="0.0.0.0", port=9081,log_level="info",workers=1,limit_concurrency=8,loop="asyncio")
