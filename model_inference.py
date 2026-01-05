
import json
import numpy as np
import re

class SimpleInference:
    def __init__(self, model_data_path):
        with open(model_data_path, "r") as f:
            data = json.load(f)
        
        self.weights = np.array(data["weights"])
        self.intercept = data["intercept"]
        self.vocabulary = data["vocabulary"]
        self.idf = np.array(data["idf"])
        self.params = data["params"]
        self.ngram_range = self.params["ngram_range"]

    def _get_ngrams(self, tokens):
        min_n, max_n = self.ngram_range
        if max_n == 1:
            return tokens
        
        all_ngrams = []
        n_tokens = len(tokens)
        for n in range(min_n, max_n + 1):
            for i in range(n_tokens - n + 1):
                all_ngrams.append(" ".join(tokens[i:i+n]))
        return all_ngrams

    def transform_and_predict(self, text):
        tokens = text.split()
        ngrams = self._get_ngrams(tokens)
        
        # Count term frequencies
        tf = np.zeros(len(self.vocabulary))
        for gram in ngrams:
            if gram in self.vocabulary:
                idx = self.vocabulary[gram]
                tf[idx] += 1
        
        if np.sum(tf) == 0:
            return 0, [0.5, 0.5]
        
        # Apply sublinear TF scaling
        if self.params["sublinear_tf"]:
            tf = np.where(tf > 0, 1 + np.log(tf), 0)
        
        # Apply IDF and build vector
        vector = tf * self.idf
        
        # L2 Normalization
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        # Predict using Logistic Regression
        score = np.dot(vector, self.weights) + self.intercept
        
        # Sigmoid
        prob_real = 1 / (1 + np.exp(-score))
        prob_fake = 1 - prob_real
        
        prediction = 1 if prob_real >= 0.5 else 0
        return prediction, [float(prob_fake), float(prob_real)]
