import os
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

# Import the correct LangChain classes
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

class FinancialSituationMemory:
    def __init__(self, name, config, embedding_model_name="models/text-embedding-004"):
        """
        Initializes memory with:
        - Deep thinking LLM (for reasoning)
        - A dedicated Gemini embedding model
        - Chroma collection for storing memories
        """
        self.config = config
        self.name = name

        # Use Gemini reasoning LLM from config
        if "deep_think_llm" in config and isinstance(config["deep_think_llm"], ChatGoogleGenerativeAI):
            self.deep_think_llm = config["deep_think_llm"]
        else:
            raise ValueError("Config must include 'deep_think_llm' as a ChatGoogleGenerativeAI object")

        # ✅ FIX 1: Use the correct class for embeddings
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model=embedding_model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        # Initialize Chroma client
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        self.situation_collection = self.chroma_client.create_collection(name=name)

    def get_embedding(self, text):
        """Get embedding for a single piece of text."""
        # ✅ FIX 2: Use the correct method for this class
        return self.embedding_model.embed_query(text)

    def add_situations(self, situations_and_advice):
        """
        Add financial situations and their corresponding advice.
        Parameter is a list of tuples (situation, recommendation)
        """
        situations = [item[0] for item in situations_and_advice]
        advice = [item[1] for item in situations_and_advice]
        ids = [str(i + self.situation_collection.count()) for i in range(len(situations))]

        # ✨ OPTIMIZATION: Embed all documents in a single, efficient API call
        embeddings = self.embedding_model.embed_documents(situations)

        self.situation_collection.add(
            documents=situations,
            metadatas=[{"recommendation": rec} for rec in advice],
            embeddings=embeddings,
            ids=ids,
        )

    def get_memories(self, current_situation, n_matches=1):
        """Find matching recommendations using Gemini embeddings."""
        query_embedding = self.get_embedding(current_situation)

        results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_matches,
            include=["metadatas", "documents", "distances"],
        )

        matched_results = []
        if results and results["documents"]:
            for i in range(len(results["documents"][0])):
                matched_results.append(
                    {
                        "matched_situation": results["documents"][0][i],
                        "recommendation": results["metadatas"][0][i]["recommendation"],
                        "similarity_score": 1 - results["distances"][0][i], # Convert distance to similarity
                    }
                )
        return matched_results


# This is the part that makes the script runnable
if __name__ == "__main__":
    # Load environment variables from a .env file
    load_dotenv()

    # Ensure your GOOGLE_API_KEY is set in your environment or a .env file
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found. Please set it in your environment or a .env file.")

    # 1. Configure the Gemini reasoning LLM (e.g., Gemini 1.5 Pro)
    gemini_reasoning = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest", # Using a powerful model for reasoning tasks
        temperature=0.7,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # 2. Initialize the memory class with the reasoning model
    # The embedding model is handled inside the class
    matcher = FinancialSituationMemory(
        name="financial_advice_memory",
        config={"deep_think_llm": gemini_reasoning},
        embedding_model_name="models/text-embedding-004" # Specify the embedding model
    )

    # 3. Add example data to the memory
    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "The housing market is cooling down and mortgage rates are at an all-time high.",
            "Re-evaluate real estate investments. Consider REITs that focus on rental properties instead of new construction."
        )
    ]
    matcher.add_situations(example_data)
    print("✅ Situations added to memory.")

    # 4. Query with a new, similar situation
    current_situation = "Market is showing crazy volatility in tech, with big investors pulling out their money."
    print(f"\n🔍 Querying with situation: '{current_situation}'")
    recommendations = matcher.get_memories(current_situation, n_matches=2)

    # 5. Print the results
    for i, rec in enumerate(recommendations, 1):
        print(f"\n--- Match {i} ---")
        print(f"Similarity Score: {rec['similarity_score']:.4f}")
        print(f"Matched Situation: {rec['matched_situation']}")
        print(f"Retrieved Recommendation: {rec['recommendation']}")