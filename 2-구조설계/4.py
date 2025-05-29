# 웹 검색 모듈 추가

# Web Search 노드
def web_search(state: GraphState) -> GraphState:
    # 검색 도구 생성
    tavily_tool = TavilySearch()

    search_query = state["question"]

    # 다양한 파라미터를 사용한 검색 예제 
    search_result = tavily_tool.search(
        query=search_query, # 검색쿼리 
        topic="news",   # 일반 주제 
        days=1,
        max_results=3,  # 최대 검색 결과
        format_output=True  # 결과 포맷팅 
    )

    return GraphState(context="\n".join(search_result))