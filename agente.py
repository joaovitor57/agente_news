import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_core.tools import Tool
from langchain_core.callbacks import BaseCallbackHandler
from textblob import TextBlob
from duckduckgo_search import DDGS  # Importação direta da biblioteca

# --- 1. Configuração Inicial ---
load_dotenv()

if os.environ.get("GOOGLE_API_KEY") is None:
    print("❌ Erro: GOOGLE_API_KEY não encontrada no arquivo .env")
    exit()

# --- 2. Monitoramento de Tokens (Requisito do PDF) ---
class TokenMonitorCallback(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):
        try:
            generations = response.generations[0]
            if generations and generations[0].generation_info:
                usage = generations[0].generation_info.get("usage_metadata", {})
                total = usage.get("total_tokens", 0)
                print(f"\n [TOKEN MONITOR] Total usado na etapa: {total}")
        except:
            pass

# --- 3. Ferramentas (Tools) ---

# Tool 1: Pesquisa de Notícias (Customizada e Direta)
def search_news_direct(query: str):
    """
    Pesquisa notícias reais no DuckDuckGo.
    Retorna os títulos, datas e resumos das últimas notícias.
    """
    print(f"\n Pesquisando notícias sobre: '{query}' no DuckDuckGo...")
    try:
        # Usamos max_results=5 para não estourar o limite de tokens
        with DDGS() as ddgs:
            # O método .news() é específico para notícias
            results = list(ddgs.news(keywords=query, region="br-pt", max_results=5))
            
            if not results:
                # Fallback: Tenta pesquisa de texto comum se news falhar
                results = list(ddgs.text(keywords=f"{query} noticias", region="br-pt", max_results=5))

            if not results:
                return "Nenhuma notícia encontrada. O servidor pode estar bloqueando a conexão."

            # Formata o resultado para o LLM ler
            formatted_results = ""
            for item in results:
                title = item.get('title', 'Sem título')
                body = item.get('body', item.get('snippet', ''))
                source = item.get('source', 'Fonte desconhecida')
                date = item.get('date', '')
                formatted_results += f"- [{date}] {title} ({source}): {body}\n"
            
            return formatted_results

    except Exception as e:
        return f"Erro crítico na ferramenta de busca: {str(e)}"

# Tool 2: Code Agent de Análise de Sentimento
def analyze_sentiment(text: str):
    """Analisa se o texto é Positivo, Negativo ou Neutro usando TextBlob."""
    print(f"\n Calculando sentimento matemática do texto...")
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.1:
        return f"POSITIVO (Score: {polarity:.2f})"
    elif polarity < -0.1:
        return f"NEGATIVO (Score: {polarity:.2f})"
    else:
        return f"NEUTRO (Score: {polarity:.2f})"

tools = [
    Tool(
        name="Pesquisar_Noticias",
        func=search_news_direct,
        description="Use para buscar as últimas notícias. Entrada: termo de busca (ex: 'Bitcoin', 'Petrobras')."
    ),
    Tool(
        name="Analisar_Sentimento",
        func=analyze_sentiment,
        description="Analisa o sentimento de um texto. Entrada: O texto/resumo da notícia encontrado."
    )
]

# --- 4. Configuração do Agente ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True,
    callbacks=[TokenMonitorCallback()]
)

# --- 5. Loop Principal ---
if __name__ == "__main__":
    print("📰 ASSISTENTE DE NOTÍCIAS (Versão Corrigida)")
    print("Dica: Se der erro de busca, tente termos mais gerais.")
    
    while True:
        user_input = input("\nTema (ou 'sair'): ")
        if user_input.lower() in ['sair', 'x']:
            break
        
        agent_executor.invoke({"input": user_input})