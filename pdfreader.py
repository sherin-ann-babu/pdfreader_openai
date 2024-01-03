from PyPDF2 import PdfReader
from dotenv import load_dotenv

#open ai embedding --- to convert texts to sequences
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS #facebook AI similarity check
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI #language model using (llmA - OPENAI)

import os

load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")

pdf_reader = PdfReader("/home/user/Documents/dl_march/Boarding_Pass(TRV-AUH)-1.pdf")

#save pdf to raw form
raw_text = ""

for i,page in enumerate(pdf_reader.pages):
    content = page.extract_text()
    if content:
        raw_text+= content
#print(raw_text)



text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size = 800,   #no.of character in one split
                chunk_overlap = 200,  #how many characters of last in next split
                length_function = len #total no.of words
)

#chunk overlap--if 3
''' this is a text
    ext of machine
    ine learning'''

texts = text_splitter.split_text(raw_text)
#print(texts)

embeddings = OpenAIEmbeddings()

document_search = FAISS.from_texts(texts,embeddings)

chain = load_qa_chain(OpenAI(),chain_type="stuff")   
#chain_type
# 1-stuff -- the text or full date is given at single time
# 2-map reduce --- each wrds passing as one by one ,used when csv loading
#3-refining ---next wrds passed only after getting result of previous wrd or charcter -- like a storage

query = "Where he arrives? "        #what information need to know

docs = document_search.similarity_search(query) #the query is passed in dcmnt search
#print(docs)

result = chain.run(input_documents = docs, question = query)
print(result)