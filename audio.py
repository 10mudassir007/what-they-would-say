from elevenlabs.client import ElevenLabs
from elevenlabs import play
import streamlit as st
from io import BytesIO


st.audio(play_audio("Hello"), format="audio/mp3")
