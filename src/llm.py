import streamlit as st
from groq import Groq


client = Groq(api_key=st.secrets["API_KEY"])

model_name = "openai/gpt-oss-120b"

def generate_answer(question, context):
    prompt = f"""
        You are a drug information assistant.
        
        You must answer the user's question ONLY using
        the provided drug documentation.

        STRICT RULES:

        1. Do not use outside knowledge.
        2. Do not invent medical information.
        3. Do not invent dosage, contraindications,
        warnings, interactions, or adverse effects.
        4. If the answer cannot be found in the provided
        documentation, say:
        "I could not find this information in the
        provided drug documentation."
        5. Keep the answer clear and concise.
        6. Every factual statement must be supported
        by the provided context.
        7. Do not create or guess page numbers.

        DOCUMENTATION:

        {context}

        USER QUESTION:

        {question}
        """

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that provides information about drugs.",
            },
            {
                "role": "user", 
                "content": prompt
            },
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content




def build_context(results):
    context = []

    for i, result in enumerate(results, start=1):
        context.append(f"""
        SOURCE{i}
        
        Document: {result['document']}

        Page: {result['page']}

        Content: {result['text']}
    """)

    return "\n".join(context)
    
