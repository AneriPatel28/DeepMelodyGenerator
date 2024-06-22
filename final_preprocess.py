import os
import json
import music21 as m21
import numpy as np
import torch
import torch.nn.functional as F
import shutil


KERN_DATASET_PATH = "/home/ubuntu/Aneri/MelodyGenerator/code/deutschl/erk" #ensure you have selected correct path.
SAVE_DIR = "dataset"
SINGLE_FILE_DATASET = "file_dataset"
MAPPING_PATH = "mapping.json"
SEQUENCE_LENGTH = 64

# durations are expressed in quarter length
ACCEPTABLE_DURATIONS = [
    0.25, # 16th note
    0.5, # 8th note
    0.75,
    1.0, # quarter note
    1.5,
    2, # half note
    3,
    4 # whole note
]

def load_songs_in_kern(dataset_path):
    songs = []
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs

def has_acceptable_durations(song, acceptable_durations):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True

# def transpose(song):
#     parts = song.getElementsByClass(m21.stream.Part)
#     measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
#     key = measures_part0[0][4]
#     if not isinstance(key, m21.key.Key):
#         key = song.analyze("key")
#     if key.mode == "major":
#         interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
#     elif key.mode == "minor":
#         interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))
#     transposed_song = song.transpose(interval)
#     return transposed_song

def transpose(song):
    # List of all major and minor keys
    major_keys = [m21.key.Key(n) for n in m21.scale.MajorScale().getPitches('C', 'B')]
    minor_keys = [m21.key.Key(n, 'minor') for n in m21.scale.MinorScale().getPitches('C', 'B')]
    all_keys = major_keys + minor_keys

    # Transpose to each key and collect results
    transposed_songs = []
    for key in all_keys:
        # Calculate interval from song's current key to the target key
        interval = m21.interval.Interval(song.analyze('key').tonic, key.tonic)
        transposed_song = song.transpose(interval)
        transposed_songs.append((transposed_song, key.name))

    return transposed_songs


def encode_song(song, time_step=0.25):
    encoded_song = []
    for event in song.flat.notesAndRests:
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi
        elif isinstance(event, m21.note.Rest):
            symbol = "r"
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")
    encoded_song = " ".join(map(str, encoded_song))
    return encoded_song

def encode_all_transposed_songs(transposed_songs):
    encoded_songs = []
    for song, key in transposed_songs:
        encoded_song = encode_song(song)
        encoded_songs.append((encoded_song, key))
    return encoded_songs


def prepare_directory(dir_path):
    # Check if the directory exists
    if os.path.exists(dir_path):
        # If it exists, remove it along with all its contents
        shutil.rmtree(dir_path)
        print(f"Removed existing directory: {dir_path}")

    # Create a new directory
    os.makedirs(dir_path)
    print(f"Created new directory: {dir_path}")

def preprocess(dataset_path):
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")
    # Prepare the save directory
    prepare_directory(SAVE_DIR)
    for i, song in enumerate(songs):
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue
        song = transpose(song)
        encoded_song = encode_all_transposed_songs(song)
        save_path = os.path.join(SAVE_DIR, str(i))
        # Saving each encoded song with its key
        for encoded_song, key in encoded_song:
            # Create a filename that includes the song index and the key
            save_path = os.path.join(SAVE_DIR, f"{i}_{key}.txt")
            with open(save_path, "w") as fp:
                fp.write(encoded_song)

        if i % 10 == 0:
            print(f"Song {i} out of {len(songs)} processed")

def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song

def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    new_song_delimiter = "/ " * sequence_length
    with open(file_dataset_path, "w") as fp_out:
        for path, _, files in os.walk(dataset_path):
            for file in files:
                file_path = os.path.join(path, file)
                with open(file_path, "r") as fp_in:
                    song = fp_in.read()
                    fp_out.write(song + " " + new_song_delimiter)
    # Remove the last delimiter if necessary
    with open(file_dataset_path, "rb+") as fp_out:
        fp_out.seek(-len(new_song_delimiter), os.SEEK_END)
        fp_out.truncate()
    return file_dataset_path

def create_mapping(file_path, mapping_path):
    mappings = {}
    vocabulary = set()

    # Read file and build vocabulary
    with open(file_path, 'r') as file:
        for line in file:
            words = line.split()
            vocabulary.update(words)

    # Create mappings from vocabulary
    for i, symbol in enumerate(sorted(vocabulary)):  # Sorting to maintain consistent order
        mappings[symbol] = i

    # Write mappings to a file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)

    print(f"Mapping created with {len(mappings)} entries.")


def convert_songs_to_int(songs):
    int_songs = []
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)
    songs = songs.split()
    for symbol in songs:
        int_songs.append(mappings[symbol])
    return int_songs


def generate_training_sequences(sequence_length, batch_size):
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)
    num_sequences = len(int_songs) - sequence_length
    vocabulary_size = len(set(int_songs))

    # Generate data in batches
    for start_idx in range(0, num_sequences, batch_size):
        end_idx = min(start_idx + batch_size, num_sequences)
        batch_inputs = []
        batch_targets = []

        for i in range(start_idx, end_idx):
            batch_inputs.append(int_songs[i:i + sequence_length])
            batch_targets.append(int_songs[i + sequence_length])

        # Convert inputs to one-hot encoding
        batch_inputs = F.one_hot(torch.tensor(batch_inputs), num_classes=vocabulary_size).float()
        batch_targets = torch.tensor(batch_targets)

        yield batch_inputs, batch_targets

# def generate_training_sequences(sequence_length, batch_size):
#     songs = load(SINGLE_FILE_DATASET)
#     int_songs = convert_songs_to_int(songs)
#     num_sequences = len(int_songs) - sequence_length
#     if num_sequences <= 0:
#         raise ValueError("Not enough sequences to train. Increase your dataset size or reduce sequence length.")
#
#     vocabulary_size = len(set(int_songs))  # Calculate once and use throughout
#     for start_index in range(0, num_sequences, batch_size):
#         end_index = min(start_index + batch_size, num_sequences)
#         batch_inputs = []
#         batch_targets = []
#         for i in range(start_index, end_index):
#             input_seq = int_songs[i:i+sequence_length]
#             target = int_songs[i+sequence_length]
#             batch_inputs.append(input_seq)
#             batch_targets.append(target)
#
#         # One hot encoding
#         batch_inputs = F.one_hot(torch.tensor(batch_inputs), num_classes=vocabulary_size).float()
#         batch_targets = torch.tensor(batch_targets)
#         yield batch_inputs, batch_targets, vocabulary_size

def main():
    preprocess(KERN_DATASET_PATH)
    file_path = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(file_path, MAPPING_PATH)
    #inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

if __name__ == "__main__":
    main()
