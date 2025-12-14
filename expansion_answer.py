import os
from pprint import pprint

import chromadb
import matplotlib.pyplot as plt
import umap
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from openai import OpenAI
from pypdf import PdfReader

from helper_utils import word_wrap, project_embeddings

# load environment variables from .env file
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)


# some example text and pdf
reader = PdfReader("data/microsoft-annual-report.pdf")
pdf_text = [page.extract_text() for page in reader.pages]


# split text into smaller chunks
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=0
)
character_split_text = character_splitter.split_text("\n\n".join(pdf_text))
token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)

# token_split_text = []
# for text in character_split_text:
#     token_split_text += token_splitter.split_text(text)

token_split_text = [chunk for text in character_split_text for chunk in token_splitter.split_text(text)]

embedding_function = SentenceTransformerEmbeddingFunction()
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(
    name="microsoft-collection",
    embedding_function=embedding_function
)


# extract the embeddings of the token_split_texts
ids = [str(i) for i in range(len(token_split_text))]
chroma_collection.add(ids=ids, documents=token_split_text)
count = chroma_collection.count()

query = "What is the total revenue for the year?"
results_one = chroma_collection.query(query_texts=[query], n_results=5)
retrieved_documents_one = results_one["documents"][0]


def augment_query_generated(inner_query, model="gpt-3.5-turbo"):
    prompt = """
            You are a helpful expert financial research assistant. 
            Provide an example answer to the given question, 
            that might be found in a document like an annual report.
            """
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": inner_query
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content


original_query = "What was the total profit for the year, and how does it compare to the previous year?"
hypothetical_answer = augment_query_generated(original_query)

joint_query = f"{original_query} {hypothetical_answer}"


results_two = chroma_collection.query(
    query_texts=joint_query, n_results=5, include=["documents", "embeddings"]
)

retrieved_documents_two = results_two["documents"][0]

embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)


retrieved_embeddings = results_two["embeddings"][0]
original_query_embedding = embedding_function([original_query])
augmented_query_embedding = embedding_function([joint_query])

projected_original_query_embedding = project_embeddings(original_query_embedding, umap_transform)
projected_augmented_query_embedding = project_embeddings(augmented_query_embedding, umap_transform)
projected_retrieved_embeddings = project_embeddings(retrieved_embeddings, umap_transform)


# Plot the projected query and retrieved documents in the embedding space
plt.figure()

plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    color="gray",
)
plt.scatter(
    projected_retrieved_embeddings[:, 0],
    projected_retrieved_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
)
plt.scatter(
    projected_original_query_embedding[:, 0],
    projected_original_query_embedding[:, 1],
    s=150,
    marker="X",
    color="r",
)
plt.scatter(
    projected_augmented_query_embedding[:, 0],
    projected_augmented_query_embedding[:, 1],
    s=150,
    marker="X",
    color="orange",
)

plt.gca().set_aspect("equal", "datalim")
plt.title(f"{original_query}")
plt.axis("off")
plt.show()  # display the plot






if __name__ == '__main__':
    # print(word_wrap(pdf_text[1], width=1000))
    # print(word_wrap(character_split_text[10], width=1000))
    #
    # print(character_split_text)
    # print(f"\ntoken_split_text: {len(token_split_text)}")
    #
    # print(embedding_function([token_split_text[10]]))
    #
    # print(count)
    #
    # for document in retrieved_documents_one:
    #     print(f"\ndocument:\n{word_wrap(document, width=1000)}")
    #
    # print(word_wrap(joint_query))
    #
    # print(word_wrap(joint_query, width=1000))
    # for doc in retrieved_documents_two:
    #     print(f"\ndocument:\n{word_wrap(doc, width=1000)}")
    #
    # pprint(results_two)

    pass