import os
from typing import List, Tuple, Dict
import re


class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".txt"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .txt file."
            )

    def load_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    with open(
                        os.path.join(root, file), "r", encoding=self.encoding
                    ) as f:
                        self.documents.append(f.read())

    def load_documents(self):
        self.load()
        return self.documents


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Define topic categories and their keywords
        self.topic_categories = {
            "EXERCISE AND MOVEMENT": [
                "exercise", "movement", "workout", "fitness", "physical activity",
                "cardio", "strength training", "flexibility", "balance", "stretching",
                "lower back pain", "neck", "shoulder", "routine", "squats", "push-ups"
            ],
            "NUTRITION AND DIET": [
                "nutrition", "diet", "eating", "meal", "food", "carbohydrates",
                "proteins", "fats", "vitamins", "minerals", "hydration", "water",
                "breakfast", "lunch", "dinner", "snack", "fiber", "probiotic"
            ],
            "SLEEP AND RECOVERY": [
                "sleep", "recovery", "insomnia", "bedtime", "sleep schedule",
                "sleep hygiene", "REM", "rest", "nap", "circadian", "melatonin"
            ],
            "STRESS MANAGEMENT AND MENTAL WELLNESS": [
                "stress", "mental wellness", "anxiety", "mindfulness", "meditation",
                "relaxation", "breathing", "grounding", "emotional", "depression",
                "worry", "overwhelmed"
            ],
            "BUILDING HEALTHY HABITS": [
                "habit", "routine", "morning", "evening", "wind-down", "formation",
                "cue", "reward", "consistency", "tracking", "progress"
            ],
            "COMMON HEALTH CONCERNS": [
                "headache", "digestive", "immune", "health concern", "illness",
                "symptom", "remedy", "treatment", "gut", "microbiome"
            ],
            "LIFESTYLE AND WELLNESS": [
                "lifestyle", "work-life balance", "social connection", "digital wellness",
                "technology", "screen time", "balance", "wellness", "well-being"
            ]
        }

    def _extract_topic_category(self, text: str) -> str:
        """Extract topic category from text based on keywords."""
        text_lower = text.lower()
        
        # Check each category for keyword matches
        category_scores = {}
        for category, keywords in self.topic_categories.items():
            score = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            if score > 0:
                category_scores[category] = score
        
        # Also check for explicit PART markers
        part_patterns = {
            "EXERCISE AND MOVEMENT": r"PART\s+1.*EXERCISE",
            "NUTRITION AND DIET": r"PART\s+2.*NUTRITION",
            "SLEEP AND RECOVERY": r"PART\s+3.*SLEEP",
            "STRESS MANAGEMENT AND MENTAL WELLNESS": r"PART\s+4.*STRESS",
            "BUILDING HEALTHY HABITS": r"PART\s+5.*HABIT",
            "COMMON HEALTH CONCERNS": r"PART\s+6.*HEALTH",
            "LIFESTYLE AND WELLNESS": r"PART\s+7.*LIFESTYLE"
        }
        
        for category, pattern in part_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return category
        
        # Return category with highest score, or default
        if category_scores:
            return max(category_scores, key=category_scores.get)
        
        return "LIFESTYLE AND WELLNESS"  # Default category

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks
    
    def split_texts_with_metadata(self, texts: List[str]) -> Tuple[List[str], List[Dict]]:
        """Split texts and return chunks with metadata."""
        chunks = []
        metadata_list = []
        
        for text in texts:
            text_chunks = self.split(text)
            chunks.extend(text_chunks)
            
            # Extract metadata for each chunk
            for chunk in text_chunks:
                topic_category = self._extract_topic_category(chunk)
                metadata_list.append({
                    "topic_category": topic_category
                })
        
        return chunks, metadata_list


if __name__ == "__main__":
    loader = TextFileLoader("data/KingLear.txt")
    loader.load()
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(loader.documents)
    print(len(chunks))
    print(chunks[0])
    print("--------")
    print(chunks[1])
    print("--------")
    print(chunks[-2])
    print("--------")
    print(chunks[-1])
