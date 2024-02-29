import dotenv
dotenv.load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_retriever_input(params):
    return params["messages"][-1].content

class LoggerStrOutputParser(StrOutputParser):
    def parse(self, text: str) -> str:
        logger.info(f"QUERY: {text}")
        return text


class Chat:
    def __init__(self):
        self.history = {}
        self.__init_chain()
        self.__init_retriever()
        self.__init_transformation_chain()
        self.retrieval_chain = (
            RunnablePassthrough.assign(
                context=self.query_transforming_retriever_chain,
            ).assign(
                answer=self.chain,
            )
        )
    
    def __init_chain(self):
        self.chat_api = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.2)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful assistant named Kianoosh who has a big brother named Mehrab. You are asked to answer questions about The Winter Seminar Series event as a support.
Winter Seminar Series (WSS which is also written as وسس in Persian) is a professional community event hosted by the Sharif University of Technology, aimed at bringing together successful Iranians globally to focus on computer science and engineering topics. Established eight years ago by the Student Scientific Chapter, WSS has become a significant four-day event where speakers present their research, share findings, and teach. The seminar includes presentations, roundtable discussions on various scientific topics, and educational workshops. These workshops are conducted online by university alumni and cover practical aspects of computer science and engineering. The event also features roundtable discussions in Persian, encouraging networking and knowledge exchange among participants.
The user has asked a question about the event. 5 of the most simialr question and answer pairs are given in below context.
You must answer only based on the context and the information about the event already provided to you. try to give positive answers about the event.
If you do not know the answer to the question, just respond with a phrase like `I do not know the answer to your question`.
rewrite the question and answer pairs in the context to match the user's question.
here is the context:

{context}""",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        self.chain = create_stuff_documents_chain(self.chat_api, prompt)
    
    def __init_retriever(self):
        loader = UnstructuredFileLoader("datasets/Q&A.txt")
        splitter = RecursiveCharacterTextSplitter(
            separators='\n\n',
            chunk_size=20,
            chunk_overlap=0,
            length_function=len,
        )
        documents = loader.load_and_split(splitter)
        vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
        loader = UnstructuredFileLoader("datasets/WSS.txt")
        documents = loader.load_and_split(splitter)
        vectorstore.add_documents(documents)
        self.retriever = vectorstore.as_retriever(k=5)
        logger.info("retriever loaded")
    
    def __init_transformation_chain(self):
        query_transform_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="messages"),
                (
                    "user",
                    """Given the above conversation, generate a search query to look up in order to get relevant information to the conversation. Only respond with the query, nothing else. give a general query that doesn't contain specific keywords. as an example:
Winter Seminal Series (WSS which is also written as وسس in Persian) is an event, so questions regarding WSS should just give queries about event without containing the specific name of the event. here are some examples of user question and query pair:
user: when is WSS being held?
query: when is event being held?
user: I'm hungry?
query: where is food being served?"""
                ),
            ]
        )
        self.query_transforming_retriever_chain = RunnableBranch(
            (
                lambda x: len(x.get("messages", [])) == 1,
                query_transform_prompt | self.chat_api | LoggerStrOutputParser() | self.retriever,
            ),
            query_transform_prompt | self.chat_api | LoggerStrOutputParser() | self.retriever,
        ).with_config(run_name="chat_retriever_chain")

    def add_user(self, user):
        self.history[user] = ChatMessageHistory()
        logger.info(f"user {user} added")

    def has_user(self, user):
        return user in self.history

    def chat(self, user, message):
        if not self.has_user(user):
            self.add_user(user)
        logger.info(f"Q: {message}")
        self.history[user].add_user_message(message)
        response = self.retrieval_chain.invoke({"messages": self.history[user].messages})
        logger.info(f"A: {response}")
        self.history[user].add_ai_message(response['answer'])
        return response['answer']

    def count_messages(self, user):
        return len(self.history[user].messages)