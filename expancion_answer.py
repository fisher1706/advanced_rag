from helper_utils import word_wrap
from pypdf import PdfReader
import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain_text_splitters  import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter


# load environment variables from .env file
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# split text into smaller chunks
reader = PdfReader("data/microsoft-annual-report.pdf")
pdf_text = [page.extract_text() for page in reader.pages]

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=0
)
character_split_text = character_splitter.split_text("\n\n".join(pdf_text))

token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)

token_split_text = []
for text in character_split_text:
    token_split_text += token_splitter.split_text(text)

# token_split_text = [chunk for text in character_split_text for chunk in token_splitter.split_text(text)]



if __name__ == '__main__':
    # print(word_wrap(pdf_text[1], width=1000))
    # print(word_wrap(character_split_text[10], width=1000))

    print(f"\ntotal chunks: {len(character_split_text)}")
    print(f"\ntoken_split_text: {len(token_split_text)}")