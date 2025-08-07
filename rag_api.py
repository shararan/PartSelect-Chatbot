from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import json
import pickle
from pathlib import Path
import logging
from dataclasses import dataclass
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
from sentence_transformers import SentenceTransformer
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class AppliancePart:
    """Represents an appliance part document"""

    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


class EmbeddingModel:
    """Handles text embeddings using sentence-transformers (local, free)"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully")

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using sentence-transformers"""
        try:
            text = str(text).replace("\n", " ").strip()
            if not text or text == "nan":
                return np.zeros(384, dtype=np.float32)

            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Error getting embedding for text: {str(e)}")
            return np.zeros(384, dtype=np.float32)

    def get_embeddings_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[np.ndarray]:
        """Get embeddings for multiple texts in batches"""
        logger.info(f"Processing {len(texts)} texts for embeddings...")

        cleaned_texts = []
        for text in texts:
            cleaned_text = str(text).replace("\n", " ").strip()
            if not cleaned_text or cleaned_text == "nan":
                cleaned_text = "No information available"
            cleaned_texts.append(cleaned_text)

        try:
            embeddings = []
            for i in range(0, len(cleaned_texts), batch_size):
                batch = cleaned_texts[i : i + batch_size]
                batch_embeddings = self.model.encode(
                    batch, convert_to_numpy=True, batch_size=batch_size
                )
                embeddings.extend([emb.astype(np.float32) for emb in batch_embeddings])

                if (i // batch_size + 1) % 10 == 0:
                    logger.info(
                        f"Processed {i + len(batch)}/{len(cleaned_texts)} texts"
                    )

            logger.info("Embedding processing complete")
            return embeddings
        except Exception as e:
            logger.error(f"Error getting embeddings batch: {str(e)}")
            return [np.zeros(384, dtype=np.float32) for _ in texts]


class AppliancePartsDB:
    """Vector database for appliance parts"""

    def __init__(self, db_path: str = "appliance_parts_db"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        self.parts: List[AppliancePart] = []
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.metadata_db_path = self.db_path / "parts_metadata.db"
        self.embeddings_path = self.db_path / "parts_embeddings.pkl"
        self.embedding_model = None

        self._init_metadata_db()

    def _init_metadata_db(self):
        """Initialize SQLite database for metadata"""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS appliance_parts (
                id TEXT PRIMARY KEY,
                part_name TEXT,
                part_id TEXT,
                mpn_id TEXT,
                part_price TEXT,
                install_difficulty TEXT,
                install_time TEXT,
                symptoms TEXT,
                product_types TEXT,
                brand TEXT,
                availability TEXT,
                product_url TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        conn.commit()
        conn.close()

    def add_part(self, part: AppliancePart):
        """Add an appliance part to the database"""
        self.parts.append(part)

        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()

        metadata = part.metadata
        cursor.execute(
            """
            INSERT OR REPLACE INTO appliance_parts 
            (id, part_name, part_id, mpn_id, part_price, install_difficulty, 
             install_time, symptoms, product_types, brand, availability, product_url, content)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                part.id,
                metadata.get("part_name", ""),
                metadata.get("part_id", ""),
                metadata.get("mpn_id", ""),
                metadata.get("part_price", ""),
                metadata.get("install_difficulty", ""),
                metadata.get("install_time", ""),
                metadata.get("symptoms", ""),
                metadata.get("product_types", ""),
                metadata.get("brand", ""),
                metadata.get("availability", ""),
                metadata.get("product_url", ""),
                part.content,
            ),
        )
        conn.commit()
        conn.close()

    def build_embeddings_matrix(self, embedding_model: EmbeddingModel):
        """Build the embeddings matrix for all parts"""
        logger.info("Building embeddings matrix...")

        texts_to_embed = []
        parts_needing_embeddings = []

        for part in self.parts:
            if part.embedding is None:
                texts_to_embed.append(part.content)
                parts_needing_embeddings.append(part)

        if texts_to_embed:
            logger.info(f"Getting embeddings for {len(texts_to_embed)} parts...")
            embeddings = embedding_model.get_embeddings_batch(texts_to_embed)

            for part, embedding in zip(parts_needing_embeddings, embeddings):
                part.embedding = embedding

        embeddings = []
        for part in self.parts:
            embeddings.append(part.embedding)

        self.embeddings_matrix = np.vstack(embeddings)

        with open(self.embeddings_path, "wb") as f:
            pickle.dump(
                {
                    "embeddings": self.embeddings_matrix,
                    "part_ids": [part.id for part in self.parts],
                },
                f,
            )

        logger.info("Embeddings matrix built and saved")

    def load_from_disk(self):
        """Load existing database from disk"""
        if not self.embeddings_path.exists():
            logger.info("No existing embeddings found")
            return

        with open(self.embeddings_path, "rb") as f:
            data = pickle.load(f)
            self.embeddings_matrix = data["embeddings"]
            part_ids = data["part_ids"]

        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        cursor.execute(
            """SELECT id, part_name, part_id, mpn_id, part_price, install_difficulty, 
                          install_time, symptoms, product_types, brand, availability, product_url, content 
                          FROM appliance_parts"""
        )
        rows = cursor.fetchall()
        conn.close()

        self.parts = []
        for row in rows:
            part_id = row[0]
            if part_id in part_ids:
                idx = part_ids.index(part_id)
                metadata = {
                    "part_name": row[1],
                    "part_id": row[2],
                    "mpn_id": row[3],
                    "part_price": row[4],
                    "install_difficulty": row[5],
                    "install_time": row[6],
                    "symptoms": row[7],
                    "product_types": row[8],
                    "brand": row[9],
                    "availability": row[10],
                    "product_url": row[11],
                }

                part = AppliancePart(
                    id=part_id,
                    content=row[12],
                    metadata=metadata,
                    embedding=(
                        self.embeddings_matrix[idx]
                        if self.embeddings_matrix is not None
                        else None
                    ),
                )
                self.parts.append(part)

        logger.info(f"Loaded {len(self.parts)} parts from disk")

    def search_semantic(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> List[AppliancePart]:
        """Search for similar parts using semantic similarity"""
        if self.embeddings_matrix is None:
            return []

        query_embedding = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]

        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            part = self.parts[idx]
            part_copy = AppliancePart(
                id=part.id,
                content=part.content,
                metadata=part.metadata.copy(),
                embedding=part.embedding,
            )
            part_copy.metadata["similarity_score"] = float(similarities[idx])
            results.append(part_copy)

        return results

    def extract_part_ids_from_query(self, query: str) -> List[str]:
        """Extract potential part IDs from a natural language query"""
        patterns = [
            r"\bPS\d+\b",
            r"\bWP\d+\b",
            r"\bW\d+\b",
            r"\b\d{8,}\b",
            r"\b[A-Z]{1,3}\d{6,}\b",
        ]

        found_ids = []
        query_upper = query.upper()

        for pattern in patterns:
            matches = re.findall(pattern, query_upper)
            found_ids.extend(matches)

        unique_ids = []
        for id_val in found_ids:
            if id_val not in unique_ids:
                unique_ids.append(id_val)

        return unique_ids

    def search_parts(self, query: str, k: int = 5) -> List[AppliancePart]:
        """Enhanced search that combines semantic search with exact ID matching"""
        logger.info(f"Searching for query: '{query}'")

        extracted_ids = self.extract_part_ids_from_query(query)
        if extracted_ids:
            logger.info(f"Extracted part IDs from query: {extracted_ids}")

        exact_matches = []
        query_upper = query.upper().strip()

        search_terms = [query_upper] + extracted_ids

        for part in self.parts:
            part_id = str(part.metadata.get("part_id", "")).upper()
            mpn_id = str(part.metadata.get("mpn_id", "")).upper()

            for search_term in search_terms:
                if search_term == part_id or search_term == mpn_id:
                    logger.info(f"Found exact match! Search term: '{search_term}'")
                    part_copy = AppliancePart(
                        id=part.id,
                        content=part.content,
                        metadata=part.metadata.copy(),
                        embedding=part.embedding,
                    )
                    part_copy.metadata["similarity_score"] = 1.0
                    if not any(
                        existing.id == part_copy.id for existing in exact_matches
                    ):
                        exact_matches.append(part_copy)
                elif len(search_term) > 3 and (
                    search_term in part_id or search_term in mpn_id
                ):
                    logger.info(f"Found partial match! Search term: '{search_term}'")
                    part_copy = AppliancePart(
                        id=part.id,
                        content=part.content,
                        metadata=part.metadata.copy(),
                        embedding=part.embedding,
                    )
                    part_copy.metadata["similarity_score"] = 0.9
                    if not any(
                        existing.id == part_copy.id for existing in exact_matches
                    ):
                        exact_matches.append(part_copy)

        if exact_matches:
            logger.info(f"Found {len(exact_matches)} exact/partial ID matches")
            if len(exact_matches) >= k:
                return exact_matches[:k]
            else:
                if self.embedding_model:
                    semantic_matches = self.search_semantic(
                        self.embedding_model.get_embedding(query),
                        k - len(exact_matches),
                    )
                    semantic_ids = {match.id for match in exact_matches}
                    additional_matches = [
                        match
                        for match in semantic_matches
                        if match.id not in semantic_ids
                    ]
                    return exact_matches + additional_matches
                else:
                    return exact_matches

        logger.info(f"No exact ID matches found, falling back to semantic search")
        if self.embedding_model:
            query_embedding = self.embedding_model.get_embedding(query)
            return self.search_semantic(query_embedding, k)
        else:
            return []


class CSVProcessor:
    """Processes appliance parts CSV file"""

    @staticmethod
    def csv_to_parts(csv_path: str) -> List[AppliancePart]:
        """Convert CSV to AppliancePart objects"""
        df = pd.read_csv(csv_path)
        parts = []

        def create_content(row):
            content_parts = []

            if pd.notna(row.get("part_name")):
                content_parts.append(f"Part: {row['part_name']}")

            part_id_columns = ["part_id", "partid", "id", "part_number", "partnumber"]
            for col in part_id_columns:
                if col in row and pd.notna(row.get(col)):
                    content_parts.append(f"Part ID: {row[col]}")
                    break

            mpn_columns = ["mpn_id", "mpn", "manufacturer_part_number", "model_number"]
            for col in mpn_columns:
                if col in row and pd.notna(row.get(col)):
                    content_parts.append(f"MPN: {row[col]}")
                    break

            if pd.notna(row.get("symptoms")):
                content_parts.append(f"Fixes symptoms: {row['symptoms']}")

            if pd.notna(row.get("product_types")):
                content_parts.append(f"Compatible with: {row['product_types']}")

            if pd.notna(row.get("brand")):
                content_parts.append(f"Brand: {row['brand']}")

            if pd.notna(row.get("install_difficulty")):
                content_parts.append(
                    f"Installation difficulty: {row['install_difficulty']}"
                )

            if pd.notna(row.get("install_time")):
                content_parts.append(f"Installation time: {row['install_time']}")

            if pd.notna(row.get("part_price")):
                content_parts.append(f"Price: {row['part_price']}")

            return " | ".join(content_parts)

        for idx, row in df.iterrows():
            content = create_content(row)

            metadata = {}
            for col in df.columns:
                value = row[col]
                metadata[col] = str(value) if pd.notna(value) else None

            part = AppliancePart(id=f"part_{idx}", content=content, metadata=metadata)
            parts.append(part)

        logger.info(f"Processed {len(parts)} appliance parts from {csv_path}")
        return parts


class AppliancePartsRAG:
    """Main RAG system for appliance parts using DeepSeek and local embeddings"""

    def __init__(self, deepseek_api_key: str, db_path: str = "appliance_parts_db"):
        self.embedding_model = EmbeddingModel()
        self.parts_db = AppliancePartsDB(db_path)
        self.parts_db.embedding_model = self.embedding_model
        self.llm_client = OpenAI(
            api_key=deepseek_api_key, base_url="https://api.deepseek.com"
        )

    def load_csv_files(self, csv_paths: list):
        """Load multiple CSV files into the database"""
        self.parts_db.load_from_disk()

        # Check if we have a complete database with all expected CSV files
        expected_files = set(csv_paths)
        loaded_files = set()

        if self.parts_db.parts:
            # Check which files are already loaded
            for part in self.parts_db.parts:
                source_file = part.metadata.get("source_file")
                if source_file:
                    loaded_files.add(source_file)

        missing_files = expected_files - loaded_files

        if not missing_files:
            logger.info(
                "All CSV files already loaded in database. Skipping CSV processing."
            )
            return
        elif loaded_files:
            logger.info(
                f"Some files already loaded. Processing missing files: {missing_files}"
            )
        else:
            logger.info("No existing database found. Processing all CSV files.")

        all_parts = []

        # Only process missing files or all files if none are loaded
        files_to_process = missing_files if missing_files else csv_paths

        for csv_path in files_to_process:
            if os.path.exists(csv_path):
                logger.info(f"Processing CSV file: {csv_path}")
                parts = CSVProcessor.csv_to_parts(csv_path)

                # Add source information to metadata
                for part in parts:
                    part.metadata["source_file"] = csv_path
                    appliance_type = (
                        "dishwasher"
                        if "dishwasher" in csv_path.lower()
                        else "refrigerator"
                    )
                    part.metadata["appliance_type"] = appliance_type

                all_parts.extend(parts)
            else:
                logger.warning(f"CSV file not found: {csv_path}")

        if not all_parts and not self.parts_db.parts:
            logger.error("No parts loaded from any CSV files")
            return
        elif all_parts:
            logger.info(f"Adding {len(all_parts)} new parts to database")

            # Add all new parts to database
            for part in all_parts:
                self.parts_db.add_part(part)

            # Rebuild embeddings matrix with all parts (existing + new)
            self.parts_db.build_embeddings_matrix(self.embedding_model)
        else:
            logger.info("Using existing parts database")

    def search_and_answer(
        self, query: str, k: int = 5, model: str = "deepseek-chat"
    ) -> Dict[str, Any]:
        """Search for relevant parts and generate an answer using DeepSeek"""
        relevant_parts = self.parts_db.search_parts(query, k=k)

        if not relevant_parts:
            return {
                "answer": "I couldn't find any relevant appliance parts for your query.",
                "sources": [],
                "query": query,
                "num_sources_found": 0,
            }

        context_parts = []

        for part in relevant_parts:
            part_info = f"""
Part Name: {part.metadata.get('part_name', 'N/A')}
Part ID: {part.metadata.get('part_id', 'N/A')}
Brand: {part.metadata.get('brand', 'N/A')}
Price: {part.metadata.get('part_price', 'N/A')}
Installation: {part.metadata.get('install_difficulty', 'N/A')} ({part.metadata.get('install_time', 'N/A')})
Symptoms it fixes: {part.metadata.get('symptoms', 'N/A')}
Compatible with: {part.metadata.get('product_types', 'N/A')}
Availability: {part.metadata.get('availability', 'N/A')}
URL: {part.metadata.get('product_url', 'N/A')}
"""
            context_parts.append(part_info.strip())

        context = "\n\n".join(context_parts)

        system_prompt = """You are a specialized appliance parts sales and support assistant. Your primary function is to provide product information and assist with customer transactions for dishwasher and refrigerator parts ONLY.

        SCOPE LIMITATIONS:
        - ONLY answer questions about dishwasher and refrigerator parts, repairs, troubleshooting, and purchasing
        - ONLY provide information based on the parts database provided to you
        - If asked about anything outside of dishwasher/refrigerator parts, politely redirect: "I specialize in dishwasher and refrigerator parts. How can I help you find the right part for your appliance?"
        - Do NOT answer questions about: other appliances, general topics, cooking, recipes, home improvement, or any non-appliance-parts related subjects

        TROUBLESHOOTING APPROACH - PRIORITY ORDER:
        1. FIRST: Provide DIY troubleshooting steps and common fixes
        2. SECOND: Suggest maintenance or cleaning procedures
        3. THIRD: Identify potential part failures and suggest replacements
        4. LAST: Recommend professional service if needed

        When customers report problems:
        - Start with "Let's troubleshoot this step by step"
        - Provide 3-5 specific diagnostic steps they can try
        - Explain what each step checks for
        - Only suggest part replacement after basic troubleshooting
        - Be encouraging: "Try these steps first - many issues can be resolved without replacement parts"

        PART RECOMMENDATIONS CONTROL:
        - You must decide whether the user is asking for specific part recommendations/suggestions
        - If user asks for recommendations, suggestions, "show me parts", "what parts", "find parts", or after troubleshooting steps don't work, respond with: SHOW_PARTS_YES
        - If user is asking general questions, needs troubleshooting help, installation guidance, or just wants information, respond with: SHOW_PARTS_NO
        - Always include either SHOW_PARTS_YES or SHOW_PARTS_NO at the very beginning of your response

        USE CASE FOCUS:
        - Troubleshooting and diagnostic guidance (PRIMARY)
        - DIY repair instructions and maintenance tips
        - Product information and specifications
        - Part identification and compatibility
        - Installation guidance and difficulty assessment
        - Pricing and availability information
        - Purchase assistance and product recommendations

        FORMATTING GUIDELINES:
        - Use **bold** for part names and important headings
        - Use bullet points (•) for lists
        - Use numbered lists (1. 2. 3.) for troubleshooting steps
        - Write clearly and professionally
        
        TECHNICAL GUIDELINES:
        
        1. SYMPTOMS FIELD UNDERSTANDING:
        The "symptoms" field represents problems that occur when this specific part is failing or broken.
        
        2. TROUBLESHOOTING FIRST APPROACH:
        - When customers report problems, ALWAYS start with troubleshooting steps
        - Check simple causes first: power, connections, settings, blockages
        - Guide through diagnostic procedures
        - Explain what normal operation should look like
        - Only mention part replacement if troubleshooting points to part failure
        
        3. PART REPLACEMENT LOGIC:
        - If troubleshooting confirms a part failure: "Based on these symptoms and our troubleshooting, the [part name] likely needs replacement"
        - Explain why the part failed and how replacement will fix the issue
        - Provide clear installation guidance
        - Suggest when to call a professional
        
        4. POSITIVE PROBLEM-SOLVING TONE:
        - Be encouraging and supportive
        - Explain that many issues have simple solutions
        - Build confidence in the customer's ability to troubleshoot
        - Use phrases like "Let's work through this together" and "This is often an easy fix"
        
        5. MAINTENANCE EDUCATION:
        - Include preventive maintenance tips
        - Explain how to avoid future problems
        - Recommend regular cleaning schedules
        - Share best practices for appliance care
        
        6. SALES ASSISTANCE (AFTER TROUBLESHOOTING):
        - Help customers identify the correct replacement part
        - Provide clear pricing and availability information
        - Guide customers to purchase links when needed
        - Suggest installation services for complex repairs
        
        7. PROFESSIONAL TONE:
        - Be helpful, knowledgeable, and patient
        - Stay within appliance parts expertise
        - Prioritize customer success over immediate sales
        - Build trust through thorough problem-solving
        
        Remember: You are a helpful repair expert first, parts salesperson second. Help customers solve problems with the least expensive solution first."""

        user_prompt = f"""Based on the following appliance parts information, please answer this question: {query}

Available Parts Information:
{context}

Please provide a helpful and detailed response based on the parts information above."""

        try:
            response = self.llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
            )

            answer = response.choices[0].message.content

            # Check if we should show parts based on AI decision
            show_parts = False
            if answer and answer.startswith("SHOW_PARTS_YES"):
                show_parts = True
                answer = answer.replace("SHOW_PARTS_YES", "").strip()
            elif answer and answer.startswith("SHOW_PARTS_NO"):
                show_parts = False
                answer = answer.replace("SHOW_PARTS_NO", "").strip()

            # Only include sources if AI decided to show parts
            sources = []
            if show_parts and relevant_parts:
                for part in relevant_parts:
                    sources.append(
                        {
                            "part_name": part.metadata.get("part_name", "N/A"),
                            "part_id": part.metadata.get("part_id", "N/A"),
                            "brand": part.metadata.get("brand", "N/A"),
                            "price": part.metadata.get("part_price", "N/A"),
                            "symptoms": part.metadata.get("symptoms", "N/A"),
                            "url": part.metadata.get("product_url", "N/A"),
                            "similarity_score": part.metadata.get(
                                "similarity_score", 0
                            ),
                        }
                    )

            # Simple cleanup to remove excessive formatting symbols
            answer = clean_formatting(answer)

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            answer = (
                f"Sorry, I encountered an error while generating the answer: {str(e)}"
            )
            sources = []

        return {
            "answer": answer,
            "sources": sources,
            "query": query,
            "num_sources_found": len(sources),
        }


def clean_formatting(text):
    """Remove formatting symbols but preserve readability"""
    if not text:
        return text

    # Remove asterisks but preserve the text structure
    text = re.sub(
        r"\*{1,}([^*\n]*?)\*{1,}", r"\1", text
    )  # Remove asterisks around text
    text = re.sub(r"\*+", "", text)  # Remove any remaining standalone asterisks

    # Remove hash symbols but preserve headers
    text = re.sub(
        r"#{1,}\s*([^\n]*)", r"\1", text
    )  # Remove hashes but keep header text

    # Improve paragraph structure
    # Add line breaks after colons for better readability
    text = re.sub(r"([^:\n]):([A-Z])", r"\1:\n\n\2", text)

    # Add line breaks before certain keywords for better structure
    text = re.sub(
        r"(What This Part Fixes|Compatibility|Installation|Where to Find)",
        r"\n\n\1",
        text,
    )
    text = re.sub(r"(Why You Might Need It|Additional|Tips|Note)", r"\n\n\1", text)

    # Fix bullet points - ensure they're on separate lines
    text = re.sub(r"•\s*([^•\n]+)", r"\n• \1", text)

    # Clean up excessive spacing but preserve paragraph breaks
    text = re.sub(r" +", " ", text)  # Multiple spaces to single space
    text = re.sub(r"\n +", "\n", text)  # Remove spaces at start of lines
    text = re.sub(r"\n{3,}", "\n\n", text)  # Max 2 line breaks

    # Ensure proper spacing after periods and colons
    text = re.sub(
        r"\.([A-Z])", r". \1", text
    )  # Space after period before capital letter
    text = re.sub(r":([A-Z])", r": \1", text)  # Space after colon before capital letter

    return text.strip()


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global RAG system
rag_system = None


def initialize_rag():
    """Initialize the RAG system"""
    global rag_system

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        logger.error("DEEPSEEK_API_KEY environment variable not set")
        return False

    logger.info("Initializing RAG system...")
    rag_system = AppliancePartsRAG(deepseek_api_key)

    # List of CSV files to load
    csv_files = ["dishwasher_parts.csv", "refrigerator_parts.csv"]

    # Check which files exist
    existing_files = []
    for csv_path in csv_files:
        if os.path.exists(csv_path):
            existing_files.append(csv_path)
            logger.info(f"Found CSV file: {csv_path}")
        else:
            logger.warning(f"CSV file not found: {csv_path}")

    if not existing_files:
        logger.error(
            "No CSV files found. Please ensure dishwasher_parts.csv and/or refrigerator_parts.csv exist."
        )
        return False

    # Load all existing CSV files
    logger.info(f"Loading {len(existing_files)} CSV files into database...")
    rag_system.load_csv_files(existing_files)
    logger.info("RAG system initialized successfully")
    return True


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    if rag_system is None:
        return (
            jsonify({"status": "error", "message": "RAG system not initialized"}),
            503,
        )

    total_parts = len(rag_system.parts_db.parts) if rag_system.parts_db.parts else 0

    return jsonify(
        {
            "status": "healthy",
            "message": "PartSelect Chat API is running",
            "total_parts": total_parts,
        }
    )


@app.route("/chat", methods=["POST"])
def chat():
    """Main chat endpoint"""
    if rag_system is None:
        return jsonify({"error": "RAG system not initialized"}), 503

    try:
        data = request.json
        message = data.get("message", "")
        conversation_history = data.get("conversation_history", [])

        if not message:
            return jsonify({"error": "Message is required"}), 400

        # Build context from conversation history
        context_messages = []
        if conversation_history:
            for msg in conversation_history[-6:]:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role and content:
                    context_messages.append(f"{role.title()}: {content}")

        context_messages.append(f"User: {message}")

        # Create enhanced query with context
        if len(context_messages) > 1:
            enhanced_query = f"""
Previous conversation context:
{chr(10).join(context_messages[:-1])}

Current question: {message}

Please answer the current question while considering the conversation context above."""
        else:
            enhanced_query = message

        result = rag_system.search_and_answer(enhanced_query)

        return jsonify(
            {
                "response": result.get(
                    "answer", "Sorry, I could not generate a response."
                ),
                "sources": result.get("sources", []),
                "num_sources_found": result.get("num_sources_found", 0),
            }
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == "__main__":
    if initialize_rag():
        print("Starting PartSelect Chat API...")
        print("API will be available at http://localhost:5000")
        print("Loaded appliance types: dishwasher and refrigerator parts")
        app.run(host="127.0.0.1", port=5000, debug=True)
    else:
        print("Failed to initialize RAG system. Please check your setup.")
        print("Make sure DEEPSEEK_API_KEY is set and CSV files exist:")
        print("- dishwasher_parts.csv")
        print("- refrigerator_parts.csv")
