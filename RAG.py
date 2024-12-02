from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

import os
import TTS

class RAG:
    def __init__(self, api_key, modelName, knowledgeStorePath='./RAGKnowledge/test.txt') -> None:
        # Prompt the user for their OpenAI API key
        self.api_key = api_key

        # Set the API key as an environment variable
        os.environ["OPENAI_API_KEY"] = self.api_key

        # Optionally, check that the environment variable was set correctly
        print("OPENAI_API_KEY has been set!")
        self.llm_model = modelName
        self.knowledgeStorePath = knowledgeStorePath

    def createVectorStore(self):
        """Create the vector store for retrieving"""
        loader = TextLoader(file_path=self.knowledgeStorePath, encoding="utf-8")
        data = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        data = text_splitter.split_documents(data)
        # use the embedding to create vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(data, embedding=embeddings)
        return vectorstore
    
    def createConversationChain(self):
        llm = ChatOpenAI(temperature=0.7, model_name="gpt-4")
        memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
        vector_store = self.createVectorStore()
        conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(),
                memory=memory
                )
        return conversation_chain

    def saveConversationToVectorStore(self, vector_store, user_message, assistant_response):
        """Save the user and assistant messages to the vector store."""
        # Prepare text to embed
        conversation_text = f"User: {user_message}\nAssistant: {assistant_response}"

        with open(self.knowledgeStorePath, "a") as knowledge:
            knowledge.write(f"this is a previous conversation between you and the user: {conversation_text}")

        # Embed the conversation text
        texts = [conversation_text]
        metadatas = [{"source": "conversation"}]

        # Add embedded conversation to vector store
        vector_store.add_texts(texts=texts, metadatas=metadatas)

    def runConversation(self, tts, user_input, conversation_chain):
        """Run a conversation loop."""
        # Generate response from the conversation chain
        assistant_response = conversation_chain({"question": user_input})['answer']

        # Display assistant's response
        print(f"Assistant: {assistant_response}")
        
        tts.convert(input_text=assistant_response)

        # Save conversation to vector store for future retrieval
        self.saveConversationToVectorStore(conversation_chain.retriever.vectorstore, user_input, assistant_response)

if __name__ == "__main__":
    key = input("Please enter your OpenAI API key: ")
    rag = RAG(api_key=key, modelName="gpt-4")
    conversation_chain = rag.createConversationChain()
    tts = TTS.TTS()
    while True:
        user_message = input("You: ")
        if user_message.lower() == "exit":
                break
        rag.runConversation(tts, user_message, conversation_chain)


