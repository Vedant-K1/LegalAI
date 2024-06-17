import iso639.language
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader, PyPDFDirectoryLoader, UnstructuredFileLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory,ConversationBufferMemory

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from deep_translator import GoogleTranslator
import iso639
from iso639 import languages
from lingua import Language, LanguageDetectorBuilder




import chainlit as cl


from langchain.embeddings import HuggingFaceBgeEmbeddings
import torch



DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DETECTOR = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()


EMBEDDINGS = HuggingFaceBgeEmbeddings(
    model_name= EMBEDDING_MODEL_NAME,
    model_kwargs={'device': DEVICE_TYPE},
    encode_kwargs={'normalize_embeddings': True}  # Set True to compute cosine similarity
)
embedding_path = "FAISS/"
vectordb = FAISS.load_local(folder_path=embedding_path, embeddings=EMBEDDINGS,allow_dangerous_deserialization=True)


from langchain.prompts import ChatPromptTemplate, PromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import LLMChain, RetrievalQA, ConversationalRetrievalChain

# def detect_source_langauge(text):
#     detected_language = str(DETECTOR.detect_language_of(text)).split('.')[1].title()
#     print('Detected Language', detected_language)
#     source_language = iso639.Language.from_name(detected_language).part1


def nyaymitra_kyr_chain(vectordb):
    llm = ChatNVIDIA(model_name="meta/llama3-70b-instruct",temperature=0.1,max_tokens=4096,api_key='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')
    system_message_prompt = SystemMessagePromptTemplate.from_template(
       """You are a law expert in India, and your role is to assist users in understanding their rights based on queries related to the provided legal context from Indian documents. Utilize the context to offer detailed responses, citing the most relevant laws and articles. If a law or article isn't pertinent to the query, exclude it. Recognize that users may not comprehend legal jargon, so after stating the legal terms, provide simplified explanations for better user understanding.
        Important Instructions:
        1. Context and Precision: Tailor your response to the user's query using the specific details provided in the legal context from India. Use only the most relevant laws and articles from the context.
        2. Comprehensive and Simplified Responses: Offer thorough responses by incorporating all relevant laws and articles. For each legal term, provide a user-friendly explanation to enhance comprehension.
        3. User-Friendly Language: Aim for simplicity in your explanations, considering that users may not have a legal background. Break down complex terms or phrases to make them more accessible to the user. Provide examples on how the law is relevant and useful to the user's query.
        LEGAL CONTEXT: \n{context}"""
    )
    human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")
    print("============HUMAN MESSAGE PROMPT============\n",human_message_prompt)
    prompt_template = ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt,
        ])  
    print("============PROMPT TEMPLATE============\n",prompt_template)
    retriever = vectordb.as_retriever()
    memory = ConversationBufferWindowMemory(k=15, memory_key="chat_history", output_key='answer', return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
      llm=llm,
      retriever=retriever,
      memory=memory,
      return_source_documents=True,
      combine_docs_chain_kwargs={"prompt": prompt_template},
    )
    return chain




@cl.on_chat_start
def start_chat():
    chain = nyaymitra_kyr_chain(vectordb)
    cl.user_session.set("chain", chain)


async def on_chat_start():
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    await chain.start_chat()



@cl.on_message
async def main(message: cl.Message):
    # # source_lang = detect_source_langauge(message.content)
    # if source_lang != 'en':
    #     trans_query = GoogleTranslator(source=source_lang, target='en').translate(message.content)
    # else:
    #     trans_query = message.content
    trans_query = message.content
    print('Translated Query', trans_query)
    await cl.Avatar(
        name="Tool 1",
        path="./public/logo_.png",
    ).send()

    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    response = await chain.acall(trans_query, callbacks=[cb])
    final_answer = response.get('answer')


    # if source_lang != 'en':
    #     trans_output = GoogleTranslator(source='auto', target=source_lang).translate(final_answer)
    # else:
    #     trans_output = final_answer

    await cl.Message(content=final_answer,author="LegalAI").send()

