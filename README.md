# Deep Learning Music Generation

## Project Overview

This project explores the intersection of artificial intelligence and music through the development of advanced deep learning models capable of generating melodies. By employing models such as Gated Recurrent Units (GRU), Variational Autoencoders (VAE), and LSTMs with Multi-Head Attention, this initiative pushes the boundaries of both technological innovation and artistic expression. Utilizing the `music21` library and PyTorch, the project aims to predict musical sequences in a time series prediction framework, treating music generation as a sequential process dependent on previous musical notes.

## Dataset

The project utilizes over 5,000 melodies from the ESAC collection, focusing on German folk songs. This dataset serves as a comprehensive repository of melodies annotated with musical and contextual metadata, providing a rich source for analysis and machine learning modeling. The ESAC dataset can be accessed at [ESAC Dataset](http://www.esac-data.org).

## Models

### Gated Recurrent Units (GRU)
- Integrates advanced neural network techniques to enhance prediction accuracy, focusing on critical parts of the input sequence.

### Variational Autoencoder (VAE)
- Employs a probabilistic approach to encode and decode high-dimensional data, facilitating the generation of new data samples that embody learned characteristics.

### LSTM with Multi Head Attention
- Uses several attention mechanisms to effectively capture the relationships between different segments of music.

## Data Preprocessing

Steps include:
1. Loading and Filtering the Dataset
2. Data Augmentation by Transposition
3. Symbolic Music Representation
4. Symbol Vocabulary Preparation
5. Time Series Data Preparation

## Results

The models demonstrated varying capabilities in generating music, with insights into the complexities of neural network-based music generation, highlighting the balance between creativity and algorithmic prediction.

## Conclusion

The project highlights the potential and challenges of using deep learning for music generation, suggesting pathways for future research in enhancing model performance and user interaction.

## How to Run

Ensure you have Python and necessary libraries installed:
- PyTorch
- music21
- MuseScore (for viewing generated scores)

To run the model training and music generation, execute the following Python scripts in order:
1. `final_preprocess.py` - For data preprocessing.
2. `final_train.py` - For training the models.
3. `final_melodygenerator.py` - For generating music based on trained models.
4. `Myapp.py` - To launch the web application interface using Streamlit.

## License

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.
