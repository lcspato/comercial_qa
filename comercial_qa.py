import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader

load_dotenv()

loader = CSVLoader(file_path="knowledge_base.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    return [doc.page_content for doc in similar_response]

llm = ChatOpenAI(temperature=1, model="gpt-4")

template = """
Você é um assistente virtual de um plano e saúde pet focado em ajudar as dúvidas do time comercial.
Sua função é auxiliar o time comercial respondendo algumas perguntas que potenciais clientes estão enviando ao time comercial.

Siga as regras abaixo:
1/ Você deve buscar se comportar com base no nosso tom de voz, para chegar ao mais próximo de como atendemos os nossos clientes.

2/ Suas respostas devem ser bem similares ao nosso tom.

Aqui está uma mensagem recebida de um novo cliente.
{message}

Esse é o nosso tom de voz.
 - Casual, Entusiasmado, Amigável, Informativo e engraçado (com moderação).
 - Quando usamos emojis, é sempre com a intenção de adicionar um toque de alegria e humanização às nossas mensagens. 🐶😊 Por exemplo, se estamos falando sobre dicas de cuidados com pets, podemos incluir um emoji para dar um ar mais leve ao assunto, mas sempre mantendo o foco na informação e no suporte confiável.
 - Nossa comunicação é projetada para fazer com que cada pessoa se sinta acolhida e compreendida, criando um ambiente onde os tutores de pets se sentem à vontade para compartilhar suas experiências e buscar conselhos. Sempre lembramos de manter a moderação e adequação dos emojis ao contexto, para que a mensagem permaneça clara e profissional. 🌟 No fim das contas, queremos que cada interação com a Petbee seja tão confortável e informativa quanto uma boa conversa com um amigo que entende e se importa com o bem-estar do seu pet. 🐱💬
 - Ser emocional e expressivo é ótimo, mas não vá muito longe com exagero ou dramatização. O humor e a sagacidade são incentivados, mas mantê-lo longe do sarcasmo ou da ofensa. Em última análise, seja informativo e útil, mantendo-se respeitoso e inclusivo - evite pandemias, estereótipos.
 - Áudios/Ligações são sempre bem-vindos em uma resolução de conflito com o cliente. A *entonação da voz geralmente é o melhor recurso para momentos onde o cliente tem um problema, descontentamento ou uma dúvida mais complexa.
 - Não podemos passar um ar corporativo, falso ou forçado.
 {best_practice}

Escreva a melhor resposta que eu deveria enviar para este potencial cliente:
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response

def main():
    st.set_page_config(
        page_title="Response manager", page_icon=":bird:") 
    st.header("Response manager")
    message = st.text_area("Pergunta do cliente")

    if message: 
        st.write("Gerando a resposta baseado nas melhores práticas...")

        result = generate_response(message)

        st.info(result)

if __name__ == '__main__':
    main()