import json
import numpy as np
import torch
import torch.nn.functional as F
import music21 as m21
from final_preprocess import SEQUENCE_LENGTH, MAPPING_PATH
from final_train import MusicModel,MusicModel2, MusicModel3


class MelodyGenerator:
    """Class to generate melodies using a pre-trained model based on LSTM or GRU."""

    def __init__(self, input_size, hidden_size, output_size, model_type):
        self.input_size = input_size  # Store input_size as an instance variable
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type

        # Conditionally initialize the correct model
        if self.model_type == 'model1':
            self.model = MusicModel(input_size, hidden_size, output_size)
            model_path = "model.pth"
        elif self.model_type == 'model2':
            self.model = MusicModel2(input_size, hidden_size, output_size)
            model_path = "model_lstm.pth"
        elif self.model_type == 'model3':
            with open(MAPPING_PATH, "r") as fp:
                self._mappings = json.load(fp)
            self.model = MusicModel3(input_size, hidden_size*2,output_size)
            model_path = "model_vae.pth"
        else:
            raise ValueError(f"Invalid model type {model_type}. Choose 'model1','model2'. or 'model3'.")

        self.model.to(self.device)  # Move model to the appropriate device
        print(f'Using device: {self.device}')
        if self.model_type == 'model1':
            try:
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.model.eval()  # Set the model to evaluation mode
            except FileNotFoundError as e:
                    print(f"Failed to load the model from {model_path}. Please check the path and try again.")
                    raise e
        elif  self.model_type == 'model2':
            try:
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.model.eval()  # Set the model to evaluation mode
            except FileNotFoundError as e:
                    print(f"Failed to load the model from {model_path}. Please check the path and try again.")
                    raise e
        elif self.model_type == 'model3':
            try:
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.model.eval()  # Set the model to evaluation mode
                    self._mappings_inverse = {v: k for k, v in self._mappings.items()}
            except FileNotFoundError as e:
                    print(f"Failed to load the model from {model_path}. Please check the path and try again.")
                    raise e
        try:
            with open(MAPPING_PATH, "r") as fp:
                self._mappings = json.load(fp)
        except FileNotFoundError as e:
            print(f"Failed to load mappings from {MAPPING_PATH}. Please check the path and try again.")
            raise e

        self._start_symbols = ["/"] * SEQUENCE_LENGTH

    def generate_melody(self, seed, num_steps, max_sequence_length, temperature=1.0):
        if self.model_type in ['model1', 'model2']:
            print(f"Seed before processing: {seed}")
            seed = seed.split()
            melody = seed[:]
            seed = self._start_symbols + seed
            seed = [self._mappings.get(symbol, "?") for symbol in seed]
            print(f"Processed seed: {seed}")
            for _ in range(num_steps):
                seed = seed[-max_sequence_length:]
                seed_tensor = torch.tensor(seed, device=self.device).long()
                onehot_seed = F.one_hot(seed_tensor, num_classes=len(self._mappings)).float()
                onehot_seed = onehot_seed.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = self.model(onehot_seed).squeeze(0)
                    probabilities = F.softmax(output.div(temperature), dim=0).cpu().numpy()
                output_int = self._sample_with_temperature(probabilities, temperature)
                seed.append(output_int)
                output_symbol = next((k for k, v in self._mappings.items() if v == output_int), None)

                print(f"Generated symbol: {output_symbol} from probability distribution")

                if output_symbol == "/":
                    break

                melody.append(output_symbol)
            print(type(melody))
            print(f"Final generated melody: {melody}")
            return melody
        else:
            seed = seed.split()
            seed = [self._mappings.get(symbol, self._mappings["_"]) for symbol in seed]
            melody1=seed
            # Initialize melody indices with valid boundary checks
            melody_indices = [idx if idx < self.input_size else self._mappings["_"] for idx in seed]

            for _ in range(num_steps):
                seed = seed[-max_sequence_length:]
                seed_tensor = torch.tensor(seed, device=self.device).long()
                onehot_seed = F.one_hot(seed_tensor, num_classes=len(self._mappings)).float()
                onehot_seed = onehot_seed.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = self.model(onehot_seed).squeeze(0)
                    probabilities = F.softmax(output.div(temperature), dim=0).cpu().numpy()

                output_int = self._sample_with_temperature(probabilities, temperature)

                # Validate output index before appending
                if output_int >= self.input_size:
                    continue

                melody_indices.append(output_int)
                melody_indices = melody_indices[-max_sequence_length:]  # Maintain fixed length

                output_symbol = self._mappings_inverse.get(output_int, "_")
                if output_symbol == "/":
                    break
            melody1 = ' '.join(self._mappings_inverse.get(idx, "_") for idx in melody_indices)
            melody1 = melody1.replace('/', '')
            return melody1

    def _sample_with_temperature(self, probabilities, temperature=1.0):
        """Sample an index from a probability array, adjusted by temperature, based on the model type."""
        if self.model_type in ['model1', 'model2']:
            # Log-transform probabilities for more control over diversity with temperature
            probabilities = np.log(probabilities + 1e-10) / temperature
            probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))
        else:
            # Flatten the array and apply clipping to avoid log of zero
            probabilities = probabilities.flatten()
            probabilities = np.clip(probabilities, 1e-10, None)
            probabilities = np.exp(np.log(probabilities) / temperature)
            probabilities /= np.sum(probabilities)

        choices = np.arange(len(probabilities))
        return np.random.choice(choices, p=probabilities)

    def save_melody(self, melody, step_duration=1, format="midi", file_name="generated_melody.mid"):
        """Save the generated melody to a MIDI file."""
        stream = m21.stream.Stream()

        if self.model_type in ['model1', 'model2']:
            # Logic for model1 and model2
            for symbol in melody:
                if symbol.isdigit():
                    pitch = m21.pitch.Pitch()
                    pitch.midi = int(symbol)
                    note = m21.note.Note(pitch.nameWithOctave, quarterLength=step_duration)
                    stream.append(note)
                elif symbol == "r":
                    rest = m21.note.Rest(quarterLength=step_duration)
                    stream.append(rest)
            stream.write(format, file_name)
        else:
            start_symbol = None
            step_counter = 1
            for i, symbol in enumerate(melody.split()):
                if symbol != "_" or i + 1 == len(melody):  # Check for sequence end or symbol change
                    if start_symbol is not None:
                        quarter_length_duration = step_duration * step_counter
                        if start_symbol == "r":
                            m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                        else:
                            m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)
                        stream.append(m21_event)
                        step_counter = 1
                    start_symbol = symbol
                else:
                    step_counter += 1

            stream.write(format, file_name)
        print(f"Melody saved to {file_name}")


if __name__ == "__main__":

    hidden_size = 128
    with open('mapping.json', 'r') as file:
        data = json.load(file)
        size = len(data)

    print("Number of top-level keys:", size)
    OUTPUT_UNITS = size
    INPUT_UNITS = size

    mg = MelodyGenerator(INPUT_UNITS, hidden_size, OUTPUT_UNITS,model_type='model3')
    seed = "67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _"
    melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.3)
    print("Generated Melody:", melody)
    mg.save_melody(melody)