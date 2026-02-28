import ollama

response = ollama.embed(
    model='nomic-embed-text',
    input='The sky is blue because of Rayleigh scattering',
)
print(response.embeddings)