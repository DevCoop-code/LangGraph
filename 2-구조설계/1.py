# ==State 정의== 
from typing import TypedDict, Annotated, List
from langchain_core.documents import Document 
import operator

class GraphState(TypedDict):
    context: Annotated[List[Document], operator.add]
    answer: Annotated[List[Document], operator.add]
    question: Annotated[str, operator.add]
    sql_query: Annotated[str, operator.add]
    binary_score: Annotated[str, operator.add]

# ==Node 정의==
def retrieve(state: GraphState) -> GraphState:
    # retrieve: 검색
    documents = "검색된 문서"
    return GraphState(context=documents)


def rewrite_query(state: GraphState) -> GraphState:
    # Query Transform: 쿼리 재작성
    documents = "검색된 문서"
    return GraphState(context=documents)


def llm_gpt_execute(state: GraphState) -> GraphState:
    # LLM 실행
    answer = "GPT 생성된 답변"
    return GraphState(answer=answer)


def llm_claude_execute(state: GraphState) -> GraphState:
    # LLM 실행
    answer = "Claude 의 생성된 답변"
    return GraphState(answer=answer)


def relevance_check(state: GraphState) -> GraphState:
    # Relevance Check: 관련성 확인
    binary_score = "Relevance Score"
    return GraphState(binary_score=binary_score)


def sum_up(state: GraphState) -> GraphState:
    # sum_up: 결과 종합
    answer = "종합된 답변"
    return GraphState(answer=answer)


def search_on_web(state: GraphState) -> GraphState:
    # Search on Web: 웹 검색
    documents = state["context"] = "기존 문서"
    searched_documents = "검색된 문서"
    documents += searched_documents
    return GraphState(context=documents)


def get_table_info(state: GraphState) -> GraphState:
    # Get Table Info: 테이블 정보 가져오기
    table_info = "테이블 정보"
    return GraphState(context=table_info)


def generate_sql_query(state: GraphState) -> GraphState:
    # Make SQL Query: SQL 쿼리 생성
    sql_query = "SQL 쿼리"
    return GraphState(sql_query=sql_query)


def execute_sql_query(state: GraphState) -> GraphState:
    # Execute SQL Query: SQL 쿼리 실행
    sql_result = "SQL 결과"
    return GraphState(context=sql_result)


def validate_sql_query(state: GraphState) -> GraphState:
    # Validate SQL Query: SQL 쿼리 검증
    binary_score = "SQL 쿼리 검증 결과"
    return GraphState(binary_score=binary_score)


def handle_error(state: GraphState) -> GraphState:
    # Error Handling: 에러 처리
    error = "에러 발생"
    return GraphState(context=error)


def decision(state: GraphState) -> GraphState:
    # 의사결정
    decision = "결정"
    return decision

# 그래프 정의
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver 
from langchain_teddynote.graphs import visualize_graph

# (1): Conventional RAG
# (2): 재검색
# (3): 멀티 LLM
# (4): 쿼리 재작성


# langgraph.graph에서 StateGraph와 END를 가져옵니다.
workflow = StateGraph(GraphState)

# ===노드를 추가===
workflow.add_node("retrieve", retrieve)

workflow.add_node("rewrite_query", rewrite_query)   # (4)

workflow.add_node("GPT 요청", llm_gpt_execute)
workflow.add_node("Claude 요청", llm_claude_execute)  # (3)
workflow.add_node("GPT_relevance_check", relevance_check)
workflow.add_node("Claude_relevance_check", relevance_check)  # (3)
workflow.add_node("결과 종합", sum_up)

# ===각 노드들을 연결===
workflow.add_edge("retrieve", "GPT 요청")
workflow.add_edge("retrieve", "Claude 요청")  # (3)
workflow.add_edge("rewrite_query", "retrieve")  # (4)
workflow.add_edge("GPT 요청", "GPT_relevance_check")
workflow.add_edge("GPT_relevance_check", "결과 종합")
workflow.add_edge("Claude 요청", "Claude_relevance_check")  # (3)
workflow.add_edge("Claude_relevance_check", "결과 종합")  # (3)

# ==조건부 엣지를 추가==
workflow.add_conditional_edges
(
    "결과 종합",    # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달 
    decision,
    {
        "재검색", "rewrite_query", # 관련성 체크 결과가 모호하다면 다시 답변을 생성
        "종료": END,  # 관련성이 있으면 종료합니다.
    }
)

# 시작점을 설정
workflow.set_entry_point("retrieve")

# 기록을 위한 메모리 저장소를 설정
memory = MemorySaver()

# 그래프를 컴파일
app = workflow.compile(checkpointer=memory)

# 그래프 시각화
visualize_graph(app)