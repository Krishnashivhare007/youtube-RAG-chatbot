from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_core.runnables import RunnablePassthrough,RunnableLambda,RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from uuid import uuid4

load_dotenv()

# Indexing 

video_id = "lj_sEsay9ro"

transcript_parts = []

full_transcript = " "

try:

    ytt_api = YouTubeTranscriptApi()

    fetched_transcript = ytt_api.fetch(video_id,languages=["en"])

    for chunk in fetched_transcript.to_raw_data():
        transcript_parts.append(f"Start: {chunk['start']}s | Duration: {chunk['duration']}s | Text: {chunk['text']}")

    full_transcript = " ".join(transcript_parts)
    
except TranscriptsDisabled:
    print("No captions available for this video.")


splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunk = splitter.create_documents([full_transcript])

embeddings = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-2')

uuids = []

for _ in range(len(chunk)):
    uuids.append(str(uuid4()))

vector_store = FAISS.from_documents(
    chunk,
    embeddings,
    ids=uuids
)

# Retriever

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":4}
)

# Augmentation

def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

parallel_chain = RunnableParallel({
    "context" : retriever | RunnableLambda(format_docs),
    "question" : RunnablePassthrough() 
})

prompt = PromptTemplate(
    template= '''
       You are a helpful assistant.
       Answer ONLY from the provided transcript context.
       If the context is insufficient, just say you don't know.
   
       {context}
       Question: {question}
   ''',
   input_variables=['context','question']
)

model = ChatGoogleGenerativeAI(model="gemini-3.7-flash")

output_parser = StrOutputParser()

chain = parallel_chain | prompt | model | output_parser

answer = chain.invoke('According to the video, what is the most important strategy for achieving a high score in the IELTS Speaking test?')

print(answer)