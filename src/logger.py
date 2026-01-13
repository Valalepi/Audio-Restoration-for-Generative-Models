import os
import shutil
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import json
from pathlib import Path

class ExperimentLogger:
    def __init__(self, base_results_dir='results_experiments'):
        self.base_dir = Path(base_results_dir)
        self.base_dir.mkdir(exist_ok=True)

    def log_experiment(self, song_name, clean_path, degraded_path, restored_path, notes="", metrics=None):
        # Sanitizza il nome del file
        safe_name = song_name.replace(' ', '_').replace('.wav', '')
        # Crea sottocartella univoca per questo test
        exp_dir = self.base_dir / safe_name
        exp_dir.mkdir(exist_ok=True)

        print(f"\n📝 Salvataggio esperimento in: {exp_dir}")

        # 1. Copia i file Audio
        shutil.copy2(clean_path, exp_dir / "01_original.wav")
        shutil.copy2(degraded_path, exp_dir / "02_degraded.wav")
        shutil.copy2(restored_path, exp_dir / "03_restored.wav")

        # 2. Genera e salva Spettrogramma
        self._save_spectrogram_plot(clean_path, degraded_path, restored_path, exp_dir / "spectrogram_comparison.png")

        # 3. Salva le Metriche (JSON + TXT)
        if metrics:
            with open(exp_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)

            with open(exp_dir / "metrics_report.txt", "w") as f:
                f.write(f"METRICS REPORT: {song_name}\n{'='*40}\n")
                f.write(f"{'METRIC':<10} | {'BASE':<8} | {'RESTORED':<8} | {'DELTA'}\n{'-'*45}\n")
                for k in ['sisdr', 'pesq', 'stoi', 'snr', 'lsd']:
                    base = metrics.get(f'{k}_baseline', 0)
                    rest = metrics.get(f'{k}_restored', 0)
                    delta = rest - base
                    f.write(f"{k.upper():<10} | {base:<8.2f} | {rest:<8.2f} | {delta:+.2f}\n")

        # 4. Salva le Note Personali
        with open(exp_dir / "my_notes.txt", "w") as f:
            f.write(f"Esperimento: {song_name}\n")
            f.write("="*30 + "\n")
            f.write(notes + "\n")

        print("✅ Salvataggio completato: Audio, Immagini, Metriche e Note.")

    def _save_spectrogram_plot(self, c, d, r, out_path):
        # Carica audio
        y_c, sr = librosa.load(c, sr=16000)
        y_d, _ = librosa.load(d, sr=16000)
        y_r, _ = librosa.load(r, sr=16000)

        # Calcola spettrogrammi
        D_c = librosa.amplitude_to_db(np.abs(librosa.stft(y_c)), ref=np.max)
        D_d = librosa.amplitude_to_db(np.abs(librosa.stft(y_d)), ref=np.max)
        D_r = librosa.amplitude_to_db(np.abs(librosa.stft(y_r)), ref=np.max)

        # Plot
        fig, ax = plt.subplots(3, 1, figsize=(10, 12))
        librosa.display.specshow(D_c, sr=sr, x_axis='time', y_axis='hz', ax=ax[0])
        ax[0].set_title('Originale (Clean)')
        librosa.display.specshow(D_d, sr=sr, x_axis='time', y_axis='hz', ax=ax[1])
        ax[1].set_title('Degradato (Input)')
        librosa.display.specshow(D_r, sr=sr, x_axis='time', y_axis='hz', ax=ax[2])
        ax[2].set_title('Ricostruito (Output)')

        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
