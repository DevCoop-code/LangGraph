from langchain_openai import ChatOpenAI  
from langchain_teddynote.evaluator import GroundednessChecker  
from langchain_teddynote.messages import messages_to_history  
from rag.utils import format_docs 

# 관련성 체크 노드
def relevance_check(state: GraphState) -> GraphState:
    # 관련성 평가기를 생성
    question_answer_relevant = GroundednessChecker(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        target="question-retrieval"
    ).create()

    # 관련성 체크를 실행("yes" or "no")
    response = question_answer_relevant.invoke(
        {"question": state["question"], "context": state["context"]}
    )

    print("==== [RELEVANCE CHECK] ====")
    print(response.score)

    return GraphState(relevance=response.score)

# 관련성 쳌 함수(router)
def is_relevant(state: GraphState) -> GraphState:
    return state["relevance"]


# Edge
workflow.add_conditional_edges(
    "relevance_check",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달 
    is_relevant,
    {
        "yes": "llm_answer",    # 관련성이 있으면 답변을 생성
        "no": "retrieve",       # 관련성이 없으면 다시 검색
    }
)