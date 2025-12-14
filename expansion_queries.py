import os

import chromadb
import matplotlib.pyplot as plt
import umap
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from openai import OpenAI
from pypdf import PdfReader

from helper_utils import project_embeddings, word_wrap

# Load environment variables from .env file
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)


# Filter the empty strings
reader = PdfReader("data/microsoft-annual-report.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]
pdf_texts = [text for text in pdf_texts if text]


# split the text into smaller chunks
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))
token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

embedding_function = SentenceTransformerEmbeddingFunction()


chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(
    "microsoft-collection", embedding_function=embedding_function
)

# extract the embeddings of the token_split_texts
ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)
chroma_collection.count()

query = "What was the total revenue for the year?"
results_one = chroma_collection.query(query_texts=[query], n_results=5)
retrieved_documents_one = results_one["documents"][0]


def generate_multi_query(inner_query, model="gpt-3.5-turbo"):
    prompt = """
            You are a knowledgeable financial research assistant. Your users are inquiring about an annual report. 
            For the given question, propose up to five related questions to assist them in finding the information they need. 
            Provide concise, single-topic questions (without compounding sentences) that cover various aspects of the topic. 
            Ensure each question is complete and directly related to the original inquiry. 
            List each question on a separate line without numbering.
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
    content = content.split("\n")
    return content

original_query = "What details can you provide about the factors that led to revenue growth?"
aug_queries = generate_multi_query(original_query)

# concatenate the original query with the augmented queries
joint_query = [original_query] + aug_queries  # original query is in a list because chroma can actually handle multiple queries, so we add it in a list

results = chroma_collection.query(
    query_texts=joint_query, n_results=5, include=["documents", "embeddings"]
)
retrieved_documents = results["documents"]

# deduplicate the retrieved documents
unique_documents = set()
for documents in retrieved_documents:
    for document in documents:
        unique_documents.add(document)


embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

# we can also visualize the results in the embedding space
original_query_embedding = embedding_function([original_query])
augmented_query_embeddings = embedding_function(joint_query)

project_original_query = project_embeddings(original_query_embedding, umap_transform)
project_augmented_queries = project_embeddings(augmented_query_embeddings, umap_transform)

retrieved_embeddings = results["embeddings"]
result_embeddings = [item for sublist in retrieved_embeddings for item in sublist]

projected_result_embeddings = project_embeddings(result_embeddings, umap_transform)

# plot the projected query and retrieved documents in the embedding space
plt.figure()
plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    color="gray",
)
plt.scatter(
    project_augmented_queries[:, 0],
    project_augmented_queries[:, 1],
    s=150,
    marker="X",
    color="orange",
)
plt.scatter(
    projected_result_embeddings[:, 0],
    projected_result_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
)
plt.scatter(
    project_original_query[:, 0],
    project_original_query[:, 1],
    s=150,
    marker="X",
    color="r",
)

plt.gca().set_aspect("equal", "datalim")
plt.title(f"{original_query}")
plt.axis("off")
plt.show()  # display the plot



if __name__ == '__main__':
    # print(word_wrap(pdf_texts[0], width=100))
    # print("*" * 100)
    # print(pdf_texts)
    #
    # print(word_wrap(character_split_texts[10]))
    # print(f"\ntotal chunks: {len(character_split_texts)}")
    #
    # print(word_wrap(token_split_texts[10]))
    # print(f"\ntotal chunks: {len(token_split_texts)}")
    #
    # print(embedding_function([token_split_texts[10]]))
    #
    # for document in retrieved_documents_one:
    #     print(word_wrap(document))
    #     print("\n")
    #
    # First step show the augmented queries
    # for query in aug_queries:
    #     print("\n", query)
    #
    # print("======> \n\n", joint_query)
    #
    # print(results)
    #
    # output the results documents
    # for i, documents in enumerate(retrieved_documents):
    #     print(f"Query: {joint_query[i]}")
    #     print("")
    #     print("Results:")
    #     for doc in documents:
    #         print(word_wrap(doc))
    #         print("")
    #     print("-" * 100)

    pass


