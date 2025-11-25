import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_core.tools import Tool
from langchain_core.callbacks import BaseCallbackHandler
from textblob import TextBlob
try:
    from duckduckgo_search import DDGS
except ImportError:
    from ddgs import DDGS

# Configuração Inicial
load_dotenv()

# Monitoramento de Tokens
class TokenMonitorCallback(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):
        """Captura e exibe o uso de tokens após cada execução do LLM."""
        try:
            generations = response.generations[0]
            if generations and generations[0].generation_info:
                usage = generations[0].generation_info.get("usage_metadata", {})
                total = usage.get("total_tokens", 0)
                print(f"\n [TOKEN MONITOR] Total tokens used in this step: {total}")
                print("-" * 50) 
        except:
            pass

# Tools

def search_news_english(query: str):
    """
    Pesquisa notícias em INGLÊS no DuckDuckGo para melhor precisão do TextBlob.
    Retorna títulos, datas e resumos.
    """
    print(f"\nSearching for news about: '{query}' (Region: US-EN)...")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(keywords=query, region="us-en", max_results=5))
            
            if not results:
                results = list(ddgs.text(keywords=f"{query} news", region="us-en", max_results=5))

            if not results:
                return "No news found."

            formatted_results = ""
            for item in results:
                title = item.get('title', 'No title')
                body = item.get('body', item.get('snippet', ''))
                source = item.get('source', 'Unknown Source')
                date = item.get('date', '')
                formatted_results += f"- [{date}] {title} ({source}): {body}\n"
            
            return formatted_results

    except Exception as e:
        return f"Critical error in search tool: {str(e)}"

def analyze_sentiment(text: str):
    """
    Tool 2 (Code Agent): Avalia se o conteúdo da notícia é negativo ou positivo.
    """
    print(f"\n Calculando sentimento do conteúdo encontrado...")
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    # Retorna string explicativa para o LLM
    if polarity > 0.1:
        return f"POLARITY SCORE: {polarity:.2f} (POSITIVE). Content is generally positive."
    elif polarity < -0.1:
        return f"POLARITY SCORE: {polarity:.2f} (NEGATIVE). Content is generally negative."
    else:
        return f"POLARITY SCORE: {polarity:.2f} (NEUTRAL). Content is neutral."

# Lista de Tools
tools = [
    Tool(
        name="Search_News",
        func=search_news_english,
        description="Useful to find the latest news about a topic. Input: search query."
    ),
    Tool(
        name="Analyze_Sentiment",
        func=analyze_sentiment,
        description="MANDATORY: Use this tool immediately after finding news. Input: The full text of the news found."
    )
]

# Configuração do Agente 

# modelo llm
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True,
    callbacks=[TokenMonitorCallback()]
)

# Loop Principal 
if __name__ == "__main__":
    print("### AGENTE AVALIADOR DE NOTÍCIAS ###")
    
    while True:
        user_input = input("\n Digite o tema da notícia (ou 'X' para sair): ")
        if user_input.lower() in ['sair', 'x']:
            break
        
        # Prompt refinado para garantir fluxo correto e resposta em PT-BR
        prompt_completo = (
            f"Please, search for the latest news about '{user_input}'. "
            f"Use the 'Search_News' tool first. "
            f"Then, YOU MUST use the 'Analyze_Sentiment' tool on the content found to get the polarity. "
            f"Finally, answer me in PORTUGUESE (PT-BR) summarizing the news and stating the sentiment found."
        )
        
        try:
            agent_executor.invoke({"input": prompt_completo})
        except Exception as e:
            print(f"Error: {e}")