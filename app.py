import streamlit as st
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
import time
import os
import json
import base64
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv

load_dotenv()

sheet_url = os.getenv("SHEET_URL")
creds_base64 = os.getenv("CREDS")  
groq_api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("MODEL_NAME")
model_provider = os.getenv("MODEL_PROVIDER")
temperature = float(os.getenv("MODEL_TEMPERATURE", 0.0))

if not sheet_url or not creds_base64 or not groq_api_key:
    st.error("Faltan variables de entorno necesarias.")
    st.stop()

creds_json = base64.b64decode(creds_base64).decode('utf-8')
with open("google_creds.json", "w") as f:
    f.write(creds_json)

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("google_creds.json", scope)
client = gspread.authorize(creds)
sheet = client.open_by_url(sheet_url).sheet1

class Classification(BaseModel):
    intents: list = Field(..., description="List of intents detected in the user's input.")
    entities: dict = Field(..., description="Dictionary of extracted entities and their values.")
    explanation: str = Field(..., description="Explanation of how the intents and entities were identified.")
    language: str = Field(..., description="Language code (ISO 639-1) of the input, e.g., 'en' or 'es'.")

def classify_input(user_input: str, intents: dict, entities: dict):
    intents_desc = "\n".join(f"- {k}: {v}" for k, v in intents.items())
    entities_desc = "\n".join(f"- {k}: {v}" for k, v in entities.items())

    prompt = ChatPromptTemplate.from_template(
        f"""
        Extract the desired information from the following passage.
        Use the following list of possible intents for classification:
        {intents_desc}
        Use the following list of possible entities to detect:
        {entities_desc}
        User input:
        {user_input}
        """
    )

    llm = init_chat_model(model_name, model_provider=model_provider, temperature=temperature, api_key=groq_api_key).with_structured_output(Classification)

    start = time.time()
    result = llm.invoke(prompt.format(user_input=user_input))
    end = time.time()

    return result.model_dump(), end - start

def log_to_gsheet(input_text, intents, entities, result, response_time):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_of_day = now.strftime("%H:%M:%S")

    row = [
        "demo", date, time_of_day,
        input_text,
        json.dumps(intents, ensure_ascii=False),
        json.dumps(entities, ensure_ascii=False),
        json.dumps(result.get("intents", []), ensure_ascii=False),
        json.dumps(result.get("entities", {}), ensure_ascii=False),
        result.get("explanation", ""),
        result.get("language", ""),
        f"{response_time:.2f}",
        model_name,
        model_provider,
        temperature
    ]

    sheet.append_row(row)

st.title("dynamic classifier")

with st.form("classification_form"):
    user_input = st.text_area("Texto del usuario", height=120)
    intents_raw = st.text_area("Intenciones (formato JSON)", value='{"saludo": "Detectar saludos", "despedida": "Detectar despedidas"}')
    entities_raw = st.text_area("Entidades (formato JSON)", value='{"nombre": "Nombre propio", "ciudad": "Nombre de una ciudad"}')
    submitted = st.form_submit_button("GO!")

if submitted:
    try:
        intents = json.loads(intents_raw)
        entities = json.loads(entities_raw)

        result, response_time = classify_input(user_input, intents, entities)
        log_to_gsheet(user_input, intents, entities, result, response_time)

        st.success(f"Clasificación completada en {response_time:.2f} segundos")
        st.subheader("Resultado")
        st.json(result)

    except Exception as e:
        st.error(f"Error: {str(e)}")

if os.path.exists("google_creds.json"):
    os.remove("google_creds.json")
