from huggingface_hub import login
import argparse


from datasets import load_dataset
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from typing import List, Any
import numpy as np

class EmbeddingModel():
    """
        An embedding model class
    """
    def __init__(self, model_checkpoint:str, type:str="hf", openai_token:str="", cohere_token:str="", device:str=""):
        self.model_checkpoint = model_checkpoint
        self.type = type

        if type == "openai":
            self.model = OpenAI(api_key=openai_token)
        elif type == "cohere":
            self.model = cohere.Client(cohere_token)
        elif type == "hf": # huggingface
            self.model = SentenceTransformer(model_checkpoint).to(device)
        else:
            raise ValueError(f"We only support openai, cohere, and hf as model_checkpoint type.")
    
    def get_openai_embedding(self, texts):
        data = self.model.embeddings.create(input = texts, model=self.model_checkpoint).data
        embeddings = []
        for obj in data:
            embeddings.append(obj.embedding)
        return embeddings

    def get_cohere_embedding(self, texts):
        response = self.model.embed(texts=texts, model=self.model_checkpoint, input_type="search_query")
        return response.embeddings

    def encode(self, texts):
        if self.type == "openai":
            embeddings = self.get_openai_embedding(texts)
        elif self.type == "cohere":
            embeddings = self.get_cohere_embedding(texts)
        else:
            embeddings = self.model.encode(texts)
        return embeddings


class DistFuse():
    """
        A DistFuse class to compute similarity scores from multiple models.
    
        e.g.,
            from distfuse import DistFuse

            model_checkpoints = [["sentence-transformers/LaBSE", "hf"], ["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "hf"]]
            weights = [1, 1]
            dist_measure = "cosine"
            model = DistFuse(model_checkpoints, weights, dist_measure=dist_measure)
            
            scores = model.score_pair(["I like apple", "I like cats"], ["I like orange", "I like dogs"])
            print(scores)
    """
    def __init__(self, model_checkpoints:List[List[str]], weights:List[float]=None, instructions:List[str]=None, dist_measure:str="euclid", openai_token:str=None, cohere_token:str=None, device:str=None):
        """
            Args:
                model_checkpoints (List[str]): a list of model checkpoints and types
                weights (List[float]): a list of weights
                instructions (List[str]): a list of instructions
                dist_measure (str): the distance measure (only accept euclidean, cosine, manhattan, by default: euclidean)
                openai_token (str): openai token
                cohere_token (str): cohere token
                device (str): device
        """
        self.model_checkpoints = model_checkpoints
        self.models = []
        self.instructions = []
        self.opt = None

        if dist_measure == "euclidean":
            self.dist_measure = euclidean_distances
            self.opt = np.min
        elif dist_measure == "cosine":
            self.dist_measure = cosine_similarity
            self.opt = np.max
        elif dist_measure == "manhattan":
            self.dist_measure = manhattan_distances
            self.opt = np.min
        else:
            raise ValueError(f"dist_measure {dist_measure} is not found.")

        for i in range(len(self.model_checkpoints)):
            model_checkpoint = self.model_checkpoints[i]
            model = EmbeddingModel(model_checkpoint[0], type=model_checkpoint[1], openai_token=openai_token, cohere_token=cohere_token, device=device)
            self.models.append(model)

            if instructions is None:
                self.instructions.append("")
            else:
                assert len(instructions) == len(self.model_checkpoints)
                self.instructions.append(instructions[i])

        if weights is None:
            self.weights = [1] * len(self.models)
        else:
            self.weights = weights

        assert len(self.models) == len(self.weights)

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'

    def score_references(self, predictions:List[str], references:List[List[str]]) -> List[float]:
        """
            Compute the scores of predictions and references. Each prediction can have a multiple references.
            Args:
                predictions (List[str]): a list of text sequences (m samples)
                references (List[List[str]]): a list of list with text sequences (m x r samples) where r is the number of references
            Returns:
                List[float]: a list of scores (m dimensions)
        """
        assert len(predictions) > 0 and len(references) > 0
        assert len(predictions) == len(references)

        scores = []
        for model_id, model in zip(range(len(self.models)), self.models):
            instruction = self.instructions[model_id]
            if instruction != "":
                instruction_predictions = []
                instruction_references = []
                for prediction, reference in zip(predictions, references):
                    instruction_predictions.append(self.get_detailed_instruct(instruction, prediction))
                    instruction_references.append(self.get_detailed_instruct(instruction, reference))
                embs1 = model.encode(instruction_predictions)
                embs2 = model.encode(instruction_references)
            else:
                embs1 = model.encode(predictions)
                embs2 = model.encode(references)
        
            scores_per_model = []
            for i in range(len(embs1)):
                reference_scores = self.dist_measure([embs1[i]], [embs2[i]])
                reference_scores = self.opt(np.array(reference_scores), axis=-1).tolist()[0]
                scores_per_model.append(reference_scores)
            scores.append(scores_per_model)

        final_scores = scores[0]
        for model_id in range(1, len(scores)):
            for j in range(len(final_scores)):
                final_scores[j] += scores[model_id][j] * self.weights[model_id]
        return final_scores


    def score_pairs(self, text_list1:List[str], text_list2:List[str]) -> List[float]:
        """
            Compute the scores of two text sequence lists
            Args:
                text_list1 (List[str]): a list of text sequences (m samples)
                text_list2 (List[str]): a list of text sequences (n samples)
            Returns:
                List[float]: a list of scores (m x n dimensions)
        """
        
        assert len(text_list1) > 0 and len(text_list2) > 0

        scores = []
        for model_id, model in zip(range(len(self.models)), self.models):
            instruction = self.instructions[model_id]
            if instruction != "":
                instruction_predictions = []
                instruction_references = []
                for prediction, reference in zip(text_list1, text_list2):
                    instruction_predictions.append(self.get_detailed_instruct(instruction, prediction))
                    instruction_references.append(self.get_detailed_instruct(instruction, reference))
                embs1 = model.encode(instruction_predictions)
                embs2 = model.encode(instruction_references)
            else:
                embs1 = model.encode(text_list1)
                embs2 = model.encode(text_list2)

            scores.append(self.dist_measure(embs1, embs2))

        final_scores = scores[0]
        for model_id in range(1, len(scores)):
            final_scores = final_scores + scores[model_id] * self.weights[model_id]
        return final_scores

def find_similar_foods(row, model, ori_df):
    df = ori_df.copy(True)
    input_desc = f"{row['Name']}. {row['Text Description']}"
    input = [input_desc] * len(df)

    # Calculate similarity scores between the input and all food descriptions
    similarity_scores = model.score_references(input, all_texts)
    print(len(similarity_scores))

    # Sort by scores to get the top 10 similar foods
    df['similarity_score'] = similarity_scores
    df = df.sort_values(by='similarity_score', ascending=False)
    top_15_similar = df[['Name', 'Text Description']].head(16)['Name'].tolist()
    top_15_similar_index = df[['Name', 'Text Description']].head(16)['Name'].index.tolist()

    # Get top 5 foods with the same fine category
    temp_df = df.copy(True)
    temp_df = temp_df[temp_df['Fine-grained categories'].apply(bool)]
    same_fine_category = temp_df[temp_df['Fine-grained categories'].apply(lambda x: set(x) == set(row['Fine-grained categories']))].head(16)['Name'].to_list()
    same_fine_category_index = temp_df[temp_df['Fine-grained categories'].apply(lambda x: set(x) == set(row['Fine-grained categories']))].head(16).index.tolist()

    # Get top 10 foods with the same coarse category
    temp_df = df.copy(True)
    temp_df = temp_df[temp_df['Coarse-grained categories'].apply(bool)]
    same_coarse_category = temp_df[temp_df['Coarse-grained categories'].apply(lambda x: set(x) == set(row['Coarse-grained categories']))].head(16)['Name'].to_list()
    same_coarse_category_index = temp_df[temp_df['Coarse-grained categories'].apply(lambda x: set(x) == set(row['Coarse-grained categories']))].head(16).index.tolist()

    return top_15_similar, top_15_similar_index, same_fine_category, same_fine_category_index, same_coarse_category, same_coarse_category_index


# Display the dataframe with the new columns
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process a specific shard of the DataFrame.")
    parser.add_argument('num_shards', type=int, help="Num shards used")
    parser.add_argument('current_shard', type=int, help="Index of the shard to process")

    # Parse arguments
    args = parser.parse_args()
    
    # Log in with your API key
    login("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")
    # Load your private dataset by its name or path in your account
    dataset = load_dataset("Exqrch/cuisine-food")
    food_df = pd.DataFrame(dataset['train'])
    
    model_checkpoints = [["intfloat/multilingual-e5-large-instruct", "hf"]]
    weights = [1]
    dist_measure = "cosine" # cosine, euclidean, manhattan
    instructions = ["Retrieve relevant food name and description"]
    embed_model_e5 = EmbeddingModel(model_checkpoints[0][0], type=model_checkpoints[0][1], openai_token="", cohere_token="", device="cuda:0")
    model = DistFuse(model_checkpoints, weights, instructions, dist_measure=dist_measure, openai_token="", cohere_token="", device="cuda:0")

    df = food_df[['Name', 'Coarse-grained categories',
        'Fine-grained categories', 'Text Description']]

    all_texts = df.apply(lambda x: f"{x['Name']}. {x['Text Description']}", axis=1).tolist()


    # Create new columns for top similar, same fine category, and same coarse category
    df['Top_15_Similar_Foods'] = None
    df['Top_15_Similar_Foods_idx'] = None
    df['Top_15_Same_Fine_Category'] = None
    df['Top_15_Same_Fine_Category_idx'] = None
    df['Top_15_Same_Coarse_Category'] = None
    df['Top_15_Same_Coarse_Category_idx'] = None
    df['embedding'] = None

    # Function to get initial and end row indices for sharding
    def get_shard_indices(df, n_shards):
        shard_size = len(df) // n_shards
        indices = []
        
        for i in range(n_shards):
            start_row = i * shard_size
            end_row = (i + 1) * shard_size if i < n_shards - 1 else len(df)  # Ensure last shard goes to the end of the DataFrame
            indices.append((start_row, end_row))
        
        return indices

    n_shards = args.num_shards
    shard_indices = get_shard_indices(df, n_shards)
    
    start_idx, end_idx = shard_indices[args.current_shard]

    from tqdm import tqdm
    # Apply the function to each row and store results in new columns
    for idx, row in tqdm(df.iterrows()):
        if start_idx <= idx < end_idx:
            top_15_similar, top_15_similar_index, same_fine_category, same_fine_category_index, same_coarse_category, same_coarse_category_index = find_similar_foods(row, model, df)
            df.at[idx, 'Top_15_Similar_Foods'] = top_15_similar
            df.at[idx, 'Top_15_Similar_Foods_idx'] = top_15_similar
            df.at[idx, 'Top_15_Same_Fine_Category'] = same_fine_category
            df.at[idx, 'Top_15_Same_Fine_Category_idx'] = same_fine_category_index
            df.at[idx, 'Top_15_Same_Coarse_Category'] = same_coarse_category
            df.at[idx, 'Top_15_Same_Coarse_Category_idx'] = same_coarse_category_index 
            df.at[idx, 'embedding'] = embed_model_e5.encode(f"{df.at[idx, 'Name']}. {df.at[idx, 'Text Description']}")

    
    df.to_csv(f"similarity_shard_{args.current_shard}", index=False)