import streamlit as st
import joblib
import pandas as pd
from PIL import Image

# Funzione per caricare il modello e lo scaler
@st.cache(allow_output_mutation=True)
def load(model_path):
    model = joblib.load(model_path)
    return model

# Funzione per eseguire l'inferenza
def inference(row, model):
    # Trasformazione dei dati di input in formato atteso dal modello
    prediction = model.predict(row)
    return prediction

# Titolo e introduzione dell'applicazione
st.title('Car Price Prediction App')
st.write('This app predicts the price of a car based on various features like brand, body type, engine type, and more.')
image = Image.open('path_to_car_image.jpg')  # Aggiungi il percorso dell'immagine
st.image(image, use_column_width=True)

# Creare un form di input per i dati dell'utente
with st.form("input_form"):
    # Campi numerici
    mileage = st.number_input('Mileage', min_value=0)
    enginev = st.number_input('Engine Volume', min_value=0.0, max_value=10.0, step=0.1)
    year = st.number_input('Year', min_value=1980, max_value=2024, step=1)
    
    # Menu a discesa per categorie
    brand = st.selectbox('Brand', ['Audi', 'BMW', 'Mercedes-Benz', 'Mitsubishi', 'Renault', 'Toyota', 'Volkswagen'])
    body = st.selectbox('Body Type', ['crossover', 'hatch', 'other', 'sedan', 'vagon', 'van'])
    engine_type = st.selectbox('Engine Type', ['Diesel', 'Gas', 'Other', 'Petrol'])
    registration = st.selectbox('Registration', ['yes', 'no'])

    submit_button = st.form_submit_button(label='Predict Price')

# Dopo che l'utente ha inviato il form
if submit_button:
    # Preparazione dei dati per il modello
    row = prepare_data(mileage, enginev, year, brand, body, engine_type, registration)
    
    # Caricare il modello
    model = load('path_to_saved_model.pkl')  # Aggiungi il percorso del modello salvato

    # Effettua la previsione
    predicted_price = inference(row, model)
    st.write(f"The predicted car price is: {predicted_price}")

# Funzione per preparare i dati per il modello
def prepare_data(mileage, enginev, year, brand, body, engine_type, registration):
    # Creare un DataFrame vuoto con tutte le colonne necessarie
    columns = ['Mileage', 'EngineV', 'Year', 'Brand_Audi', 'Brand_BMW', 'Brand_Mercedes-Benz', 'Brand_Mitsubishi',
               'Brand_Renault', 'Brand_Toyota', 'Brand_Volkswagen', 'Body_crossover', 'Body_hatch', 'Body_other', 
               'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Diesel', 'Engine Type_Gas', 'Engine Type_Other', 
               'Engine Type_Petrol', 'Registration_no', 'Registration_yes']
    data = pd.DataFrame(columns=columns)
    data.loc[0] = [0] * len(columns)

    # Impostare i valori numerici e categorici
    data.at[0, 'Mileage'] = mileage
    data.at[0, 'EngineV'] = enginev
    data.at[0, 'Year'] = year
    data.at[0, f'Brand_{brand}'] = 1
    data.at[0, f'Body_{body}'] = 1
    data.at[0, f'Engine Type_{engine_type}'] = 1
    data.at[0, f'Registration_{registration}'] = 1
    
    return data
############################################################################################
import streamlit as st
import joblib
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler

# Funzione per caricare il modello e lo scaler
@st.cache(allow_output_mutation=True)
def load(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Funzione per eseguire l'inferenza
def inference(row, model, scaler):
    # Applicare lo scaling solo alle colonne 'Mileage' e 'EngineV'
    scaled_features = scaler.transform(row[['Mileage', 'EngineV']])
    row[['Mileage', 'EngineV']] = scaled_features

    # Trasformazione dei dati di input in formato atteso dal modello
    prediction = model.predict(row)
    return prediction

# Titolo e introduzione dell'applicazione
st.title('Car Price Prediction App')
st.write('This app predicts the price of a car based on various features like brand, body type, engine type, and more.')
image = Image.open('path_to_car_image.jpg')  # Aggiungi il percorso dell'immagine
st.image(image, use_column_width=True)

# Creare un form di input per i dati dell'utente
with st.form("input_form"):
    # ... (stesso codice per la creazione dei campi di input)

    submit_button = st.form_submit_button(label='Predict Price')

# Dopo che l'utente ha inviato il form
if submit_button:
    # Preparazione dei dati per il modello
    row = prepare_data(mileage, enginev, year, brand, body, engine_type, registration)
    
    # Caricare il modello e lo scaler
    model, scaler = load('path_to_saved_model.pkl', 'path_to_saved_scaler.pkl')  # Aggiungi i percorsi corretti

    # Effettua la previsione
    predicted_price = inference(row, model, scaler)
    st.write(f"The predicted car price is: {predicted_price}")

# Funzione per preparare i dati per il modello (rimane invariata)
def prepare_data(mileage, enginev, year, brand, body, engine_type, registration):
    # ... (stesso codice per la preparazione dei dati)
