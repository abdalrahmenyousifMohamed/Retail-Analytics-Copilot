import os
import re
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimpleRetriever:
    def __init__(self, docs_dir: str):
        self.docs_dir = docs_dir
        self.chunks = []
        self.vectorizer = None
        self.vectors = None
        self._load_documents()
    
    def _load_documents(self):
        """Load and chunk documents."""
        for filename in os.listdir(self.docs_dir):
            if filename.endswith('.md'):
                filepath = os.path.join(self.docs_dir, filename)
                with open(filepath, 'r') as f:
                    content = f.read()
                
                # Simple chunking by paragraphs/sections
                sections = re.split(r'\n#{1,3}\s+', content)
                for idx, section in enumerate(sections):
                    if section.strip():
                        self.chunks.append({
                            'id': f"{filename.replace('.md', '')}::chunk{idx}",
                            'content': section.strip(),
                            'source': filename
                        })
        
        # Build TF-IDF vectors
        if self.chunks:
            texts = [chunk['content'] for chunk in self.chunks]
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.vectors = self.vectorizer.fit_transform(texts)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve top-k relevant chunks."""
        if not self.chunks:
            return []
        
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.vectors)[0]
        
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                **self.chunks[idx],
                'score': float(scores[idx])
            })
        
        return results
