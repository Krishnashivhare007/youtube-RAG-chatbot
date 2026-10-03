from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from uuid import uuid4

load_dotenv()

# Indexing

video_id = "lj_sEsay9ro" # only id not full url

transcript_parts = []
try:

    ytt_api = YouTubeTranscriptApi()

    fetched_transcript = ytt_api.fetch(video_id,languages=["en"])

    for chunk in fetched_transcript.to_raw_data():
        transcript_parts.append(f"Start: {chunk['start']}s | Duration: {chunk['duration']}s | Text: {chunk['text']}")


    full_transcript = " ".join(transcript_parts)
    
    

except TranscriptsDisabled:
    print("No captions available for this video.")


splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
chunks = splitter.create_documents([full_transcript])

# print(len(chunks))

# print(chunks[0])

embeddings = GoogleGenerativeAIEmbeddings(
    model='models/gemini-embedding-2'
)

uuids = []

for _ in range(len(chunks)): #_ ka matlab hai "mujhe loop variable ki value ki zarurat nahi hai."
    uuids.append(str(uuid4()))

vector_store = FAISS.from_documents(
    chunks,
    embeddings,
    ids=uuids
)

# print(vector_store.index_to_docstore_id)

# Retrieval

retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k":4})

retriever.invoke('According to the video, what is the most important strategy for achieving a high score in the IELTS Speaking test?')

# Augmentation 

llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash")

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

question = 'According to the video, what is the most important strategy for achieving a high score in the IELTS Speaking test?'
retrieved_docs = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({'context':retrieved_docs,'question':question})

# print(final_prompt)

# Generation

answer = llm.invoke(final_prompt)

# print(answer.content[0]['text'])



