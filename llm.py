from typing import Any, Iterator

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def get_llm(model="gpt-4o", streaming: bool = False):
    return ChatOpenAI(model=model, streaming=streaming)


def get_dictionary_chain():
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    dictionary_prompt = ChatPromptTemplate.from_template(
        f"""사용자의 질문을 보고 우리의 사전을 참고해서 사용자의 질문을 변경해주세요. 만약 변경할 필요가 없다고 판단된다면, 
        사용자의 질문을 변경하지 않아도 됩니다. 사전 : {dictionary}

        질문 : {{question}}
        """
    )
    return dictionary_prompt | get_llm() | StrOutputParser()


# RAG 답변 스타일·계산 방식을 보여 주는 few-shot (맥락 인용, 추측 금지)
RAG_FEW_SHOT_EXAMPLES = [
    {
        "input": "과세표준 1,200만원인 거주자의 종합소득 산출세액은 어떻게 되나요?",
        "output": "맥락의 세율표에 따르면 1,400만원 이하 구간은 과세표준의 6퍼센트입니다. "
        "따라서 1,200만원 × 6% = 72만원입니다.",
    },
    {
        "input": "종합소득 과세표준이 5,000만원이면 세율은 어떻게 적용되나요?",
        "output": "맥락상 1,400만원 초과 5,000만원 이하 구간에 해당합니다. "
        "84만원 + (5,000만원 − 1,400만원) × 15%로 계산합니다.",
    },
    {
        "input": "문서에 없는 임의의 공제율을 써서 세액을 줄여도 되나요?",
        "output": "아니요. 맥락에 나온 세율·공식만 사용해야 합니다. "
        "맥락에 없는 공제는 적용할 수 없다고 답해야 합니다.",
    },
]


def _get_rag_few_shot_prompt() -> FewShotChatMessagePromptTemplate:
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    return FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=RAG_FEW_SHOT_EXAMPLES,
    )


def get_retriever():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)
    index_name = "tax-markdown-index"
    database = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )
    return database.as_retriever(search_kwargs={"k": 4})

def get_history_retriever():
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        get_llm(), get_retriever(), contextualize_q_prompt
    )
    return history_aware_retriever


def get_conversational_rag_chain(*, stream_answer: bool = False):
    llm_answer = get_llm(streaming=stream_answer)
    few_shot_prompt = _get_rag_few_shot_prompt()

    qa_system = (
        "역할: 소득세 관련 조력자입니다.\n"
        "규칙:\n"
        "- 아래 '참고 맥락'에 있는 내용만 근거로 답하세요.\n"
        "- 맥락에 없는 사항은 추측하지 말고 모른다고 말하세요.\n"
        "- 세액·구간 계산이 필요하면 맥락의 세율표·공식을 단계적으로 적용해 설명하세요.\n"
        "- 이어지는 예시는 답변 형식·톤의 참고용입니다."
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"{qa_system}\n\n참고 맥락:\n{{context}}"),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = get_history_retriever()
    question_answering_chain = create_stuff_documents_chain(llm_answer, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answering_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


def _answer_to_str(answer) -> str:
    if answer is None:
        return ""
    if isinstance(answer, str):
        return answer
    content = getattr(answer, "content", None)
    if isinstance(content, str):
        return content
    if hasattr(content, "text"):
        return str(content.text)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return "".join(parts)
    if hasattr(answer, "text"):
        return str(answer.text)
    return str(answer)


def _tax_chain(stream_answer: bool):
    return (
        {"query": get_dictionary_chain()}
        | RunnablePassthrough.assign(input=lambda x: x["query"])
        | get_conversational_rag_chain(stream_answer=stream_answer)
    )


def _yield_answer_deltas(chunks: Iterator[Any]) -> Iterator[str]:
    accumulated = ""
    for chunk in chunks:
        if not isinstance(chunk, dict) or "answer" not in chunk:
            continue
        full = _answer_to_str(chunk["answer"])
        if not full:
            continue
        if full.startswith(accumulated):
            delta = full[len(accumulated) :]
            accumulated = full
        else:
            delta = full
            accumulated += full
        if delta:
            yield delta


def stream_ai_response(
    user_message: str, session_id: str = "default"
) -> Iterator[str]:
    load_dotenv(dotenv_path="../inflearn-llm-application/.env")
    chain = _tax_chain(stream_answer=True)
    stream = chain.stream(
        {"question": user_message},
        config={"configurable": {"session_id": session_id}},
    )
    yield from _yield_answer_deltas(stream)


def get_ai_response(user_message: str, session_id: str = "default") -> str:
    load_dotenv(dotenv_path="../inflearn-llm-application/.env")
    chain = _tax_chain(stream_answer=False)
    result = chain.invoke(
        {"question": user_message},
        config={"configurable": {"session_id": session_id}},
    )
    if isinstance(result, dict) and "answer" in result:
        return _answer_to_str(result["answer"])
    return _answer_to_str(result)
