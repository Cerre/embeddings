import pandas as pd
from annoy import AnnoyIndex

# from dotenv import load_dotenv
from openai import OpenAI
import os

# Load the API key from an .env file
# openai.api_key = os.getenv('OPENAI_API_KEY')
OpenAI.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

from create_embeddings import generate_embedding

# Load the JSON data into a pandas DataFrame
df = pd.read_json("combined_transcriptions_with_embeddings.json")

# # If your JSON structure is deeply nested or pandas doesn't automatically normalize the nested structures into a flat table,
# # you might need to manually normalize the data or extract the embeddings in a separate step.

# Initialize lists to hold your data
embeddings = []
video_ids = []
text_chunks_info = []  # For storing the text and timestamps

for _, row in df.iterrows():
    for chunk in row["text_chunks"]:
        embeddings.append(chunk["embedding"])
        video_ids.append(row["videoId"])
        text_chunks_info.append((chunk["start_time"], chunk["chunk_text"]))


f = len(embeddings[0])  # Dimensionality of the embeddings
t = AnnoyIndex(f, "angular")  # Using Angular distance

for i, vec in enumerate(embeddings):
    t.add_item(i, vec)

t.build(10)  # 10 trees for a balance between precision and performance
t.save("vector_database.ann")


# Example query: let's use the first embedding as a query
text = "I swear could have done better in one v one  history, an absolute upset, both players should be embarrassed. complete failure"
# text = "zen once again scores against mawkzy. It's an incredibly tied game"
# text = "spree zenzo played him look how fast noo is in the recovery though he's got not a lot of boost to play with though here double Bump by Zen and that one lands as well just like the final game of zet versus V it's turning into an air dribble bump 1 V one six goals a piece who's going to come out on top in this physical battle almost a dead ball kickoff they both get a second chance at it which Zen wins and he's got of course another setup for an aable here not the easiest position to bump nle from Zen instead Ops for the 50/50 KN might dis possessive here Zen Dives anyway wow that's aggressive I he's looking to keep the shooting threat open here and no once again closes the Gap just so quickly Den's "
# text = "disgusting, absolutely dusgusting"
# text = "I swear we could have done better one of the most embarrassing moments in One V one history keeping the ball up and just a complete failure from both players"
# Load the indexed file
u = AnnoyIndex(f, "angular")
u.load("vector_database.ann")  # Load your Annoy index
# Example query using a generated embedding
query_embedding = generate_embedding(
    client, text
)  # Assuming generate_embedding returns the correct embedding
# query_embedding = embeddings[4357]

# Finding the 10 nearest neighbors
n_neighbors = 3
nearest_ids = u.get_nns_by_vector(query_embedding, n_neighbors)
print(nearest_ids)
# Retrieve information for the nearest neighbors
nearest_info = [
    (video_ids[i], text_chunks_info[i][0], text_chunks_info[i][1]) for i in nearest_ids
]

for video_id, timestamp, text in nearest_info:
    print(f"Video ID: {video_id}, Timestamp: {timestamp}, Text: {text}\n")
