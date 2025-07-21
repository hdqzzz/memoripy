# OpenRouter example:

import os
from memoripy import MemoryManager, JSONStorage
from memoripy.implemented_models import AliyunChatModel, AliyunEmbeddingModel

def main():
    # Replace 'your-api-key' with your actual API key for the chat completions service you use
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key or api_key == "your-api-key":
        raise ValueError("Please set your API key.")

    # Define chat and embedding models
    chat_model_name = "qwen-plus" # Specific chat model name
    embedding_model_name = "text-embedding-v4"  # Specific embedding model name

    # Choose your storage option
    storage_option = JSONStorage("interaction_history.json")
    # Or use in-memory storage:
    # from memoripy import InMemoryStorage
    # storage_option = InMemoryStorage()

    # Initialize the MemoryManager with the selected models and storage
    memory_manager = MemoryManager(AliyunChatModel(api_key, chat_model_name), AliyunEmbeddingModel(api_key, embedding_model_name), storage=storage_option)

    # New user prompt
    new_prompt = "We are building a new AI agent"

    # Load the last 5 interactions from history (for context)
    short_term, _ = memory_manager.load_history()
    last_interactions = short_term[-5:] if len(short_term) >= 5 else short_term

    # Retrieve relevant past interactions, excluding the last 5
    relevant_interactions = memory_manager.retrieve_relevant_interactions(new_prompt, exclude_last_n=5)

    # Generate a response using the last interactions and retrieved interactions
    response = memory_manager.generate_response(new_prompt, last_interactions, relevant_interactions)

    # Display the response
    print(f"Generated response:\n{response}")

    # Extract concepts for the new interaction
    combined_text = f"{new_prompt} {response}"
    concepts = memory_manager.extract_concepts(combined_text)

    # Store this new interaction along with its embedding and concepts
    new_embedding = memory_manager.get_embedding(combined_text)
    memory_manager.add_interaction(new_prompt, response, new_embedding, concepts)

if __name__ == "__main__":
    main()
