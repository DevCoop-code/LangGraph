# LangGraph
LangGraph를 활용한 Agent LLM을 Ollama를 이용해 Local에서 실행시키는 프로젝트 

## Requirement
- Python 3.11.11
- [Ollama](https://ollama.com/) 설치 필요 
- ``$ pip install -r requirements.txt``

Agent 활용을 위한 ToolCalling은 지원하는 AI 모델이 따로 있음
## Ollama Tool Calling 지원 모델 리스트
- [모델 리스트](https://ollama.com/search?c=tools)

리스트 확인 후 Ollama로 모델 다운로드 
- ``$ ollama pull [다운받고자 하는 모델명]``
    - ex] ``ollama pull llama3.1``

## Reference
- [LangChain Tool Calling 지원모델 리스트](https://python.langchain.com/docs/integrations/chat/)
- [TeddyNote ToolCalling Agent](https://github.com/teddylee777/langchain-kr/blob/main/15-Agent/03-Agent.ipynb)
- [TeddyNote LangChain](https://wikidocs.net/265663)

## Ollama Reference
- Download 받은 Model 확인 명령어
    - ``$ ollama list``