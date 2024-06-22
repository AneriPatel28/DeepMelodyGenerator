
""""

Importing Required Modules

"""

import streamlit as st
from final_melodygenerator import MelodyGenerator
import torch
import os
from music21 import environment, stream, note, pitch, converter, tempo
import subprocess

import json


SEQUENCE_LENGTH = 64


def apply_dark_theme():
    st.set_page_config(page_title="Melody Generator", layout="wide")

    # Define dark theme CSS with targeted adjustments for space reduction
    dark_theme_css = """
    <style>
    html {
        --primary: #000000;
        --background-color: #002b36;
        --main-text-color: #ffffff;
        --secondary-text-color: #ffffff;  /* White color */
        --main-color: #073642;
        --color: #ffffff;
    }
    body {
        margin: 0;
        padding: 0;
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .stApp {
        padding-top: 0 !important;  /* Remove padding at the top of the app */
    }
    header {
        display: none;  /* Hides the Streamlit header completely */
    }
    .block-container {
        padding: 1rem 1rem;  /* Adjust overall padding inside the app */
    }
    .stButton>button {
        color: var(--main-text-color);
        background-color: var(--primary);
    }
    .stTextInput>div>div>input, .stTextArea>textarea, .stTextInput .st-bd, .stTextArea .st-bd {
        background-color: var(--main-color);
        color: var(--secondary-text-color);
    }
    .st-bj {
        background-color: var(--main-color);
    }
    /* Additional styles for subheaders */
    .stSubheader {
        background-color: var(--main-color)
        color: var(--secondary-text-color);
    }
    /* Additional styles to reduce top spacing */
    .css-1adrfps {
        padding-top: 0px;
    }
    .stMarkdown {
        background-color: var(--main-text-color);
        color: var(--secondary-text-color);
    }
    </style>
    """


    st.markdown(dark_theme_css, unsafe_allow_html=True)

apply_dark_theme()


# Set device for model computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize environment for MuseScore path
env = environment.Environment()
env['musescoreDirectPNGPath'] = '/usr/bin/musescore'

# Define global setting
hidden_size = 128# Adjust this to your actual sound font path
with open('mapping.json', 'r') as file:
    data = json.load(file)
    size = len(data)  # If the JSON is an object, this returns the number of top-level keys.

print("Number of top-level keys:", size)
output_size = size
input_size = size
# Set title


st.title("Welcome to the Melody Generator! 🎵")

def midi_number_to_note_name(midi_number):
    """Converts MIDI number to a music21 note name, returns None if it's a rest placeholder."""
    if midi_number.lower() == 'r':  # Check if the input is 'r' for rest
        return None  # Return None for rests to handle them separately in the calling function
    p = pitch.Pitch()
    p.midi = int(midi_number)  # Convert to integer if it's not 'r'
    return p.nameWithOctave

def text_to_score(text_melody, bpm=120):
    """Converts a space-separated string of MIDI numbers, 'r' for rests, and '_' for continuation into a music21 score."""
    s = stream.Score(id='mainScore')
    p = stream.Part(id='part')
    s.append(p)

    # Set the tempo
    mm = tempo.MetronomeMark(number=bpm)
    s.insert(0, mm)  # Insert the tempo mark at the beginning of the score

    melody_elements = text_melody.split()  # Split the string into components
    last_element = None

    for element in melody_elements:
        if element == '_':
            if last_element and isinstance(last_element, note.Note):
                # Extend the previous note
                last_element.quarterLength += 0.25  # Extend by quarter note duration
            elif last_element and isinstance(last_element, note.Rest):
                # Extend the previous rest
                last_element.quarterLength += 0.25
        elif element == 'r':
            # Add a rest
            rest = note.Rest(quarterLength=0.25)
            p.append(rest)
            last_element = rest
        else:
            note_name = midi_number_to_note_name(element)
            if note_name:
                n = note.Note(note_name, quarterLength=0.25)
                p.append(n)
                last_element = n

    return s

def convert_midi_to_wav(midi_file_path, output_file_path, sound_font):
    command = [
        'fluidsynth',
        '-ni',
        sound_font,
        midi_file_path,
        '-F',
        output_file_path,
        '-r',
        '44100'
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Error: {result.stderr.decode()}")
        return False
    return True


# Define path to your sound font file
sound_font_path = '/home/ubuntu/generating-melodies-with-rnn-lstm/GRU/soundfont.sf2'

# Function to delete files if they exist
env = environment.Environment()
env['musescoreDirectPNGPath'] = '/usr/bin/musescore'
# Set up the environment for MuseScore

def delete_file_if_exists(file_path):
    """Removes the file if it exists."""
    if os.path.exists(file_path):
        os.remove(file_path)

def save_score_image_with_xvfb(score, file_name="score.png"):
    """Uses xvfb and MuseScore to convert music XML to an image."""
    temp_musicxml = file_name.replace('.png', '.musicxml')
    score.write('musicxml', fp=temp_musicxml)
    delete_file_if_exists(file_name)
    command = ['xvfb-run', '-a', 'musescore', '-o',file_name, temp_musicxml]
    subprocess.run(command, check=True)
    delete_file_if_exists(temp_musicxml)

# Helper function to delete files if they exist
def delete_file_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

# Convert MIDI to WAV
def convert_midi_to_wav(midi_file_path, output_file_path, sound_font):
    command = ['fluidsynth', '-ni', sound_font, midi_file_path, '-F', output_file_path, '-r', '44100']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        st.error(f"Error converting MIDI to WAV: {result.stderr.decode()}")
        return False
    return True

# Generate and display sheet music as an image
def save_score_image_with_xvfb(score, file_name="score.png"):
    temp_musicxml = file_name.replace('.png', '.musicxml')
    score.write('musicxml', fp=temp_musicxml)
    delete_file_if_exists(file_name)
    subprocess.run(['xvfb-run', '-a', 'musescore', '-o', file_name, temp_musicxml], check=True)
    delete_file_if_exists(temp_musicxml)




# Define function to handle melody generation
def handle_melody_generation(seed_text, model_choice):
    if seed_text:
        if model_choice == 'GRU':
            model_type = 'model1'
        elif model_choice == 'LSTM_attention':
            model_type = 'model2'
        elif model_choice == 'Vae':
           model_type= 'model3'

        melody_generator = MelodyGenerator(input_size, hidden_size, output_size, model_type=model_type)
        melody_generator.model.to(device)
        melody = melody_generator.generate_melody(seed_text, 500, SEQUENCE_LENGTH, 0.3)
        print("jiejifjeif",melody)
        st.subheader("Generated Melody")
        if model_type == 'model1' or model_type == 'model2':
            generated_melody_str = ' '.join([str(m) for m in melody])
            st.text_area("", generated_melody_str, height=150)
            melody_generator.save_melody(melody, file_name="generated_melody.mid")
        else:
            melody.replace('/', '')
            st.text_area("", melody, height=150)

        # Save and display melody

        convert_midi_to_wav("generated_melody.mid", "generated_melody.wav", sound_font_path)

        if os.path.exists("generated_melody.wav"):
            with open("generated_melody.wav", "rb") as wav_file:
                st.audio(wav_file.read(), format='audio/wav')
        else:
            st.error("Failed to convert MIDI to WAV.")

        # Display and download MIDI

        st.download_button("Download MIDI", data=open("generated_melody.mid", "rb").read(),
                           file_name="generated_melody.mid", mime="audio/midi")

        # Generate score image
        if model_type=='model1' or model_type=='model2':
            score = text_to_score(generated_melody_str)
        else:
            score = text_to_score(melody)
        save_score_image_with_xvfb(score)
        st.subheader("Music Sheet")
        st.image("score-1.png", caption="Generated Sheet Music")


# Streamlit UI


seed_text = st.text_input("Enter a seed melody (notes separated by spaces):",
                          value="67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _")
model_choice = st.radio("Choose the model type:", ( 'GRU', 'LSTM_attention','Vae',),horizontal=True)


if st.button('Generate Melody'):
    handle_melody_generation(seed_text, model_choice)