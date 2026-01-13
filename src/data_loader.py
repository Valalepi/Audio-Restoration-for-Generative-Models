import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from pathlib import Path

class AudioDataset(Dataset):
    """
    Gestisce il caricamento e il pre-processing dei file audio.
    Eredita da torch.utils.data.Dataset.
    """
    def __init__(self, clean_dir, degraded_dir, sr=16000, length=48000):
        self.sr = sr         # Sampling Rate (16kHz standard)
        self.length = length # Lunghezza fissa in samples (48000 = 3 secondi)

        # Cerca tutti i file .wav nella cartella 'clean'
        self.clean_files = sorted(list(Path(clean_dir).glob('*.wav')))
        self.degraded_files = []

        # Per ogni file pulito, cerca il corrispondente degradato
        for cf in self.clean_files:
            # Costruisce il path atteso: clean_dir/../degraded/nomefile.wav
            # Nota: Path(degraded_dir) deve essere il percorso completo alla cartella degraded
            df = Path(degraded_dir) / cf.name

            # Verifica che esista, altrimenti avvisa
            if df.exists():
                self.degraded_files.append(df)
            else:
                print(f"Warning: File degradato mancante per {cf.name}")

        # Stampa di debug per capire se ha trovato i file
        if len(self.clean_files) == 0:
            print(f"⚠️ ATTENZIONE: Nessun file wav trovato in {clean_dir}")

    def __len__(self):
        """Restituisce il numero totale di campioni nel dataset"""
        return len(self.clean_files)

    def __getitem__(self, idx):
        """
        Carica l'i-esimo campione dal disco
        Chiamata automaticamente dal DataLoader in loop.
        """
        # Caricamento audio con Librosa (normalizza automaticamente tra -1 e 1)
        # Gestiamo eventuali errori di lettura file
        try:
            clean, _ = librosa.load(self.clean_files[idx], sr=self.sr)
            degraded, _ = librosa.load(self.degraded_files[idx], sr=self.sr)
        except Exception as e:
            print(f"Error loading file {idx}: {e}")
            # Ritorna tensori di zeri in caso di file corrotto per non bloccare il training
            return torch.zeros(1, self.length), torch.zeros(1, self.length)

        # --- GESTIONE LUNGHEZZA FISSA ---
        # Voglio batch di dimensioni identiche.

        # CASO 1: Audio troppo corto -> Aggiungiamo zeri alla fine (Padding)
        if len(clean) < self.length:
            pad = self.length - len(clean)
            clean = np.pad(clean, (0, pad))
            degraded = np.pad(degraded, (0, pad))

        # CASO 2: Audio troppo lungo -> Tagliamo l'eccedenza (Cropping)
        # Qui prendiamo solo i primi 'length' samples per semplicità
        else:
            clean = clean[:self.length]
            degraded = degraded[:self.length]

        # Conversione in Tensori PyTorch
        # Unsqueeze(0) aggiunge la dimensione del canale: [Length] -> [1, Length]
        # È necessario perché Conv1d si aspetta [Batch, Canali, Tempo]
        clean_t = torch.FloatTensor(clean).unsqueeze(0)
        deg_t = torch.FloatTensor(degraded).unsqueeze(0)

        return deg_t, clean_t


def get_loaders(base_path, batch_size=16):
    """
   Carica i dati da cartelle già separate (train/val) su disco,
    rispettando lo split manuale fatto in fase di preparazione dati.

    Struttura attesa per ogni cartella "dataset" che viene passata:
      /train/clean
      /train/degraded
      /val/clean
      /val/degraded
    """
    train_dir = Path(base_path) / 'train'
    val_dir = Path(base_path) / 'val'

    print(f"🔍 Looking for training data in: {train_dir}")

    # Istanzia Dataset separati per Train e Val
    train_ds = AudioDataset(
        clean_dir=train_dir / 'clean',
        degraded_dir=train_dir / 'degraded'
    )

    val_ds = AudioDataset(
        clean_dir=val_dir / 'clean',
        degraded_dir=val_dir / 'degraded'
    )

    print(f"✅ Found {len(train_ds)} training samples and {len(val_ds)} validation samples.")

    # Crea i DataLoader
    #num_workers=0 perché colab aveva problemi a utilizzarne di più su Drive
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    # In validazione, metto Shuffle = False per avere risultati consistenti
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader
