import os
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from groq import InternalServerError,APIConnectionError
import time
import os
import uuid
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO

from dotenv import load_dotenv
import streamlit as st
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["ELEVEN_API_KEY"] = os.getenv("ELEVEN_API_KEY")

tool = TavilySearch(
    max_results=10,
    search_type='basic'
)

def search_web(query):
  """Extract Data From Tavily Search"""
  return tool.invoke(query)

# template = """
# You are a Q&A chatbot that generates responses based on the specified character’s voice. Your task is to answer the user's query in a way that reflects the character's known speech, behavior, and worldview.

# Use the context provided to ensure your answer is relevant. Incorporate the character's statements into your response where applicable. Try to keep the response succinct for simple questions, without unnecessary elaboration, but do not omit important details when the query requires a more nuanced response, make the response like the character himself is speaking.

# Start responding to just the query, don't mention any other details about the character
# - **Character’s tone and style**: Maintain consistency with the character's tone (e.g., serious, humorous, formal, informal) as it aligns with the statements provided and make it as the character themselves is narrating, the data is diverse make the response sound like only the mentioned quotes in the statements.
# - **Response goal**: Make sure the answer is directly related to the user query, while staying true to the character’s perspective and personality.
# - **Context**: Use context to help guide the response and keep it relevant to the user’s question.
# - **Language**: Respond to the query in the language specified by the user,
# - **History**: Use the history to access previous conversations and change the words in the answers according to history while maintaining the character's tone.

# Query: {query},\n
# Context: {context},\n
# User-specified character's statements: {statements},
# Language specified by the user: {language},
# History: {history}
# """


template = """
You are a Q&A chatbot that generates responses based on the specified character’s voice. Your task is to answer the user's query in a way that reflects the character's known speech, behavior, and worldview.

You must adapt your response style based on the character’s previous replies. Use history to maintain continuity in tone, personality, and style — especially in how you phrase and structure responses. If the new question is similar to a past one, reflect a consistent speaking style or phrasing, while adjusting content as needed.

For simple, factual queries, avoid robotic or generic answers. Even if the query is mathematical or direct, express the answer with the character’s known speech patterns, and in a way consistent with how they previously answered similar things. Think like the character — how *would they* explain it again?

Do NOT explain the character. Just answer as if the character is speaking.

Guidelines:
- **Tone and Style**: Match the character’s tone (e.g., serious, humorous, philosophical) based on their actual quotes and the tone used in previous responses.
- **Context Use**: Use relevant context to inform the answer.
- **History Influence**: Use history to shape *how* you answer, not just *what* you answer. Maintain consistency in expression.
- **Language**: Respond in the language specified by the user.
- **Succinctness**: Be concise for direct questions, elaborate only when depth is required.

Query: {query}
Context: {context}
Character's Quotes and Statements: {statements}
Language: {language}
Conversation History (use this to match tone and phrasing): {history}
"""

prompt_template = ChatPromptTemplate.from_template(template)


query = "Explain Time Dilation"


def get_data(query:str, character:str):
    """Get Data from Tavily ( The info about the topic and statements by the person/character)"""
    context = search_web(query)

    statements = search_web(f"Exact quotes, speeches and statements by {character}")
    if 'results' not in statements:
        return context,statements
    statements = '\n\n'.join([statements['results'][x]['content']for x in range(len(statements))])

    return context, statements

def get_response(query:str, character:str,model:str = "llama-3.1-8b-instant", temp:float = 0, lang:str = "English",history:str=""):
    context, statements = get_data(query,character)

    llm = ChatGroq(model=model,
               temperature=temp,timeout=60)
    
    chain = prompt_template | llm
    
    #response = 
    try:
        return chain.invoke({"query":query,"statements":statements,"context":context,"language":lang,"history":history}).content
    except InternalServerError:
       return "Server error, Please select a different model"
    except APIConnectionError:
        return "Server error, Please select a different model"
def generator(string: str):
   for i in string:
      yield i + ""
      time.sleep(0.01)

def generate_audio(text: str, voice_id: str = "JBFqnCBsd6RMkjVDRZzb", model_id: str = "eleven_flash_v2_5") -> str:
    """Generate an MP3 audio file from text using ElevenLabs and return file path."""
    try:
        client = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))
        audio = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=model_id,
            output_format="mp3_44100_128",
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=1.0,
                style=0.5,
                use_speaker_boost=False,
                speed=1.0,
            ),
        )

        file_name = f"{uuid.uuid4()}.mp3"
        with open(file_name, "wb") as f:
            for chunk in audio:
                if chunk:
                    f.write(chunk)
        return file_name
    except Exception as e:
        st.error("Eleven Labs API Rate limit exceeded")


def play_and_cleanup_audio(file_path: str):
    """Play the given MP3 file and delete it afterward."""
    try:
        audio = AudioSegment.from_file(file_path, format="mp3")
        play(audio)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"✅ Deleted temp file: {file_path}")


st.title("What they would say")

language = st.sidebar.selectbox("Select Language",["English","Urdu"])

list_models = ["llama-3.1-8b-instant","llama-3.3-70b-versatile","meta-llama/llama-4-scout-17b-16e-instruct","meta-llama/llama-4-maverick-17b-128e-instruct","deepseek-r1-distill-llama-70b","qwen-qwq-32b","mistral-saba-24b","gemma2-9b-it"]

model = st.sidebar.selectbox("Select LLM",list_models)

character = st.sidebar.text_input("Character",placeholder="e.g. Albert Einstein")

temp = st.sidebar.slider(label="Temperature",min_value=0.,
    max_value=2.,
    value=0.,  
    step=0.01)
clear_history = st.sidebar.button("Clear History")

if 'history' not in st.session_state or clear_history:
    st.session_state.history = []

if len(st.session_state.history) > 10:
    st.session_state.history = st.session_state.history[1:]

st.markdown(f"#### Selcted Character: {character}")
    
for i in st.session_state.history[-10:]:
        st.chat_message('user').write(i['Query'])
        st.chat_message('assistant').write(i['Response'])
query = st.chat_input("Ask me")
if query:
    st.chat_message('user').write(query)
    #history.append(f"Query : {query}\n")
    history = "\n".join(
        [f"User: {h['Query']}\n{character.title()}: {h['Response']}" for h in st.session_state.history[-10:]]
    )
    response = get_response(query,character.lower(),model,temp,language.lower(),history)
    st.session_state.history.append({"Query":query,"Response":response})
    
    st.chat_message('assistant').write_stream(generator(response))

    if st.sidebar.button("🔊 Play Last Response"):
        if st.session_state.history:
            last_response = st.session_state.history[-1]["Response"]
            with st.spinner("Generating audio..."):
                audio_path = generate_audio(last_response)
                play_and_cleanup_audio(audio_path)
        else:
            st.sidebar.warning("No response available to play.")
