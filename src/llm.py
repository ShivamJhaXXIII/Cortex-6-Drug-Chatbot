import streamlit as st
from groq import Groq


client = Groq(api_key=st.secrets["API_KEY"])

model_name = "openai/gpt-oss-120b"

def generate_answer(question, context):
    prompt = f"""
        You are a drug information assistant.

        Answer the user's question ONLY using the
        provided documentation.

        Rules:

        1. Do not use outside knowledge.
        2. Do not invent medical information.
        3. If the documentation does not contain
        enough information, say so clearly.
        4. Do not invent citations or page numbers.
        5. Do not mention SOURCE numbers.
        6. Give a concise, factual answer.
        7. If relevant, mention important limitations
        stated in the documentation.

        DOCUMENTATION:

        {context}

        QUESTION:

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
    
