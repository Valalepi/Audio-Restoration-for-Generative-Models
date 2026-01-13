import torch
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from src.models import LightweightUNet

class AudioRestorer:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🔊 Loading model on {self.device}...")

        # Carica architettura e pesi
        self.model = LightweightUNet().to(self.device)

        # Gestione caricamento sicuro (map_location evita errori se carichi su CPU un modello allenato su GPU)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model.eval() #spegne BatchNorm/Dropout per l'inferenza (in questo caso, abbiamo solo BatchNorm)
        print("✅ Model loaded successfully.")

    def restore_file(self, input_path, output_path, chunk_size=48000, overlap=4000):
        """
        Restaura un file audio intero processandolo a pezzi (chunks).
        Usa overlap-add per evitare 'click' tra un pezzo e l'altro.
        """
        # Carica audio
        y, sr = librosa.load(input_path, sr=16000)

        # Prepara buffer di output e buffer per contare le sovrapposizioni
        #servono come contenitori per resturare la canzone dopo aver sistemato i chunk
        y_out = np.zeros_like(y)
        count = np.zeros_like(y)

        #per non dare la canzone intera al modello, la dividiamo in chunk che vengono uniti tra loro
        # Finestra di Hanning per ammorbidire i bordi dei chunk (cross-fade).

        # Overlap (Sovrapposizione): Invece di fare [0-3s], [3-6s], facciamo [0-3s], [2.5-5.5s]. I chunk si accavallano.
        # Hanning Window (Finestra a Campana): È una funzione matematica che assomiglia a una collina (parte da 0, sale a 1, scende a 0).
        window = np.hanning(chunk_size)

        # Loop sui chunk con passo (stride) minore della lunghezza per sovrapporre (es. [0-3s], [2.5-5.5s]).
        stride = chunk_size - overlap

        for start in range(0, len(y), stride):
            end = start + chunk_size

            # Gestione ultimo pezzetto (padding se serve)

            # Calcola quanto manca (chunk_size - len(chunk) = 44.000).
            # Aggiunge 44.000 zeri alla fine (np.pad). Ora il chunk è lungo 48.000.
            # Il modello lo processa tranquillamente.

            chunk = y[start:end]
            if len(chunk) < chunk_size:
                pad_len = chunk_size - len(chunk)
                chunk_padded = np.pad(chunk, (0, pad_len))
            else:
                chunk_padded = chunk
                pad_len = 0

            # Inferenza sul chunk
            with torch.no_grad():
                t_in = torch.FloatTensor(chunk_padded).unsqueeze(0).unsqueeze(0).to(self.device)
                t_out = self.model(t_in)
                cleaned_chunk = t_out.squeeze().cpu().numpy()

            # Rimuovi padding se c'era
            # Alla fine, tagliamo via quegli zeri aggiunti (cleaned_chunk[:-pad_len])
            #per tornare alla lunghezza originale di 4.000 e incollare il pezzo finale pulito.
            if pad_len > 0:
                cleaned_chunk = cleaned_chunk[:-pad_len]
                real_len = len(cleaned_chunk)
            else:
                real_len = chunk_size

            # Overlap-Add pesato dalla finestra
            # (Usiamo la finestra solo se il chunk è intero, altrimenti finestra adattata o niente finestra sui bordi estremi)
            # Per semplicità qui sommiamo diretto con normalizzazione media

            y_out[start:start+real_len] += cleaned_chunk
            count[start:start+real_len] += 1.0

        # Normalizza dove ci sono state sovrapposizioni
        count[count == 0] = 1.0 # Evita div per zero
        y_out = y_out / count

        # Salva
        sf.write(output_path, y_out, sr)
        return output_path
