import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from typing import List
import numpy as np
import torch
import argparse
from tqdm import tqdm

import os
from utils import RESOURCE_DIR

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
            embeddings = self.model.encode(texts, batch_size=256, show_progress_bar=True)
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
    
    def score_cross_references(self, references:List[List[str]]) -> List[float]:
        """
            Compute the 
        """
        scores = []
        for model_id, model in zip(range(len(self.models)), self.models):
            instruction = self.instructions[model_id]
            if instruction != "":
                instruction_references = []
                for reference in zip(references):
                    instruction_references.append(self.get_detailed_instruct(instruction, reference))
                embs = model.encode(instruction_references)
            else:
                embs = model.encode(references)
        
            scores_per_model = []
            for i in tqdm(range(len(embs))):
                for j in range(len(embs)):
                    reference_scores = self.dist_measure([embs[i]], [embs[j]])
                    reference_scores = self.opt(np.array(reference_scores), axis=-1).tolist()[0]
                    scores_per_model.append(reference_scores)
            scores.append(scores_per_model)

        final_scores = scores[0]
        for model_id in range(1, len(scores)):
            for j in range(len(final_scores)):
                final_scores[j] += scores[model_id][j] * self.weights[model_id]
        return embs, final_scores
    
    def only_embeds(self, references:List[List[str]]) -> List[float]:
        """
            Compute the 
        """
        for model_id, model in zip(range(len(self.models)), self.models):
            instruction = self.instructions[model_id]
            if instruction != "":
                instruction_references = []
                for reference in zip(references):
                    instruction_references.append(self.get_detailed_instruct(instruction, reference))
                embs = model.encode(instruction_references)
            else:
                embs = model.encode(references)

        return embs


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


def calculate_pairwise_similarities(df, model):
    all_texts = [f"{row['Name']}. {row['Text Description']}" for _, row in df.iterrows()]

    # Call the model to get similarity scores
    embs, similarity_scores = model.score_cross_references(all_texts)
    similarity_matrix = np.array(similarity_scores).reshape(len(df), len(df))

    # Create a DataFrame from the similarity matrix
    torch_dict_embd = {name: embed for name, embed in zip(df['Name'].to_list(), embs)}
    similarity_df = pd.DataFrame(similarity_matrix, index=df['Name'], columns=df['Name'])

    return torch_dict_embd, similarity_df

def generate_embeddings(food_df_path, similarity_save_path, embeddings_save_path):
    food_df = pd.read_csv(food_df_path)
    
    model_checkpoints = [["intfloat/multilingual-e5-large-instruct", "hf"]]
    weights = [1]
    dist_measure = "cosine" # cosine, euclidean, manhattan
    instructions = ["Retrieve relevant food name and description"]
    model = DistFuse(model_checkpoints, weights, instructions,
                     dist_measure=dist_measure, openai_token="",
                     cohere_token="", device="cuda:0")

    # Just filter to trim down required 
    torch_dict_embd, similarity_df = calculate_pairwise_similarities(food_df, model)
    
    # Save embedding and similarity to parquet for compression
    torch.save(torch_dict_embd, embeddings_save_path)
    similarity_df.to_parquet(similarity_save_path, compression='gzip')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Arguments for experiment regressor mode
    parser.add_argument('-f', '--food_df_path', type=str, required=False, default=os.path.join(RESOURCE_DIR, "food_df.csv"), help="Path used to load food dataframe")
    parser.add_argument('-s', '--similarity_save_path', type=str, required=False, default=os.path.join(RESOURCE_DIR, "similarity.parquet"), help="Path used to save the similarity matrix")
    parser.add_argument('-e', '--embeddings_save_path', type=str, required=False, default=os.path.join(RESOURCE_DIR, "embeddings_dict.pt"), help="Path used to save the embeddings")
    args = parser.parse_args()
    
    os.makedirs(RESOURCE_DIR, exist_ok=True)
    
    generate_embeddings(args.food_df_path, args.similarity_save_path, args.embeddings_save_path)