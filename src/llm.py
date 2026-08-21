import streamlit as st
from groq import Groq


client = Groq(api_key=st.secrets["API_KEY"])

model_name = "openai/gpt-oss-120b"

def generate_answer(question, context, history=None):
    history = history or []
    recent_history = history[-6:]
    conversation = "\n".join(
        f"{message['role'].upper()}: {message['content']}"
        for message in recent_history
    )

    prompt = f"""
        You are a drug information assistant.

        Answer the user's question ONLY using the
        provided documentation.

        You may have information from multiple documents.

        Keep information from different documents separate when necessary.
        When comparing products or drugs, identify which document each fact comes from.
        Use only the provided documentation.
        Do not combine facts from different products unless the documents support that comparison.
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

        If the question asks for a comparison between
        multiple documents or products:

        1. Identify the relevant evidence from each source.
        2. Keep facts associated with their correct source.
        3. Compare the sources explicitly.
        4. Do not let evidence from one source stand in for another source.
        5. If information is available for only one source, clearly state that the other source was not found.

        DOCUMENTATION:

        {context}

        CONVERSATION HISTORY:

        {conversation or "No previous conversation."}

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


def rewrite_query(question, history, groq_client, model):
    """
    Convert a follow-up question into a standalone
    question suitable for document retrieval.
    """

    if not history:
        return question

    recent_history = history[-6:]

    conversation = "\n".join(
        f"{message['role'].upper()}: "
        f"{message['content']}"
        for message in recent_history
    )

    prompt = f"""
    You rewrite follow-up questions into standalone search queries.

    Conversation:
    {conversation}

    Current question:
    {question}

    Rules:

    1. If the current question is already standalone,
    return it unchanged.
    2. If it refers to something from the conversation
    using words like "it", "this", "that", "they",
    "what about", or "how about", resolve the reference.
    3. Preserve the exact drug/product name when known.
    4. Do not answer the question.
    5. Return ONLY the rewritten question.

    Standalone question:
    """

    response = groq_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()

def build_context(results):
    context = []

    for i, result in enumerate(results, start=1):
        context.append(f"""
        SOURCE{i}
        
        Document: {result['document']}

        Page: {result['page']}

        Section: {result.get('section', 'Unknown')}

        Content: {result['text']}
    """)

    return "\n\n-------------------\n\n".join(context)
    
