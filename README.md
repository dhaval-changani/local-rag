# local-rag

Retrieval Augmented Generation running on a local modal

User Question
     |
     v
[1. EMBED the question]        <-- HuggingFace (converts text to numbers)
     |
     v
[2. SEARCH your documents]     <-- Your code (find most similar chunks)
     |
     v
[3. BUILD a prompt]            <-- Your code (question + found context)
     |
     v
[4. SEND to Ollama Gemma3:4b]  <-- Ollama (generates the answer)
     |
     v
Answer
