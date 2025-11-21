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

os.environ["LANGCHAIN_TRACING_V2"] = "false"

if os.environ.get("GOOGLE_API_KEY") is None:
    print("Erro: GOOGLE_API_KEY não encontrada no arquivo .env")
    exit()

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
        except:
            pass

# Tools

def search_news_english(query: str):
    """
    Pesquisa notícias em INGLÊS no DuckDuckGo.
    Retorna títulos, datas e resumos.
    """
    print(f"\nSearching for news about: '{query}' (Region: US-EN)...")
    try:
        with DDGS() as ddgs:
            # MUDANÇA: region="us-en" força notícias globais/EUA em inglês
            results = list(ddgs.news(keywords=query, region="us-en", max_results=5))
            
            if not results:
                # Fallback para busca de texto se news falhar
                results = list(ddgs.text(keywords=f"{query} news", region="us-en", max_results=5))

            if not results:
                return "No news found. The server might be blocking the connection."

            # Formata o resultado para o LLM ler
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
    Analisa se o texto é Positivo, Negativo ou Neutro usando TextBlob.
    Funciona perfeitamente para textos em Inglês.
    """
    print(f"\n Calculando sentimento")
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    # Classificação baseada na polaridade (-1 a 1)
    if polarity > 0.1:
        return f"POSITIVO (: {polarity:.2f})"
    elif polarity < -0.1:
        return f"NEGATIVO (: {polarity:.2f})"
    else:
        return f"NEUTRO (: {polarity:.2f})"

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
        description="MANDATORY: Use this tool immediately after finding news to calculate the polarity score. Input: The full text of the news."
    )
]

# Configuração do Agente 

# MODELO E VERSAO DA LLM USADA
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
#  Configuração do Agente
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True,
    callbacks=[TokenMonitorCallback()]
)

# Loop Principal 
if __name__ == "__main__":
    print("NEWS SENTIMENT ANALYST ( )")
    print("Example: 'COP30', 'Formula 1', 'Apple'")
    
    while True:
        user_input = input("\n Tema de busca ou 'X' para encerrar: ")
        if user_input.lower() in ['sair', 'x']:
            break
        
        # Prompt Injection: Forçamos o agente a seguir o fluxo correto em inglês
        prompt_completo = (
            f"Search for the latest news about '{user_input}'. "
            f"Then, YOU MUST use the 'Analyze_Sentiment' tool "
            f"on the content of the news found to give me the polarity score."
        )
        
        try:
            agent_executor.invoke({"input": prompt_completo})
        except Exception as e:
            print(f"Error: {e}")
