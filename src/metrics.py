import torch
import torchmetrics.audio as ta
from torchmetrics.audio import ShortTimeObjectiveIntelligibility, PerceptualEvaluationSpeechQuality
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio
import numpy as np
import librosa

class MetricsCalculator:
    def __init__(self, device='cpu'):
        self.device = device
        # Inizializziamo le metriche pesanti
        # PESQ Wideband (16kHz) è lo standard per la qualità vocale/musicale
        try:
            self.pesq = PerceptualEvaluationSpeechQuality(fs=16000, mode='wb').to(device)
            self.stoi = ShortTimeObjectiveIntelligibility(fs=16000).to(device)
            print("✅ Metriche PESQ e STOI caricate su GPU/CPU.")
        except Exception as e:
            print(f"⚠️ Attenzione: Impossibile caricare PESQ/STOI ({e}). Verranno saltate.")
            self.pesq = None
            self.stoi = None

    def calculate_all(self, clean_path, degraded_path, restored_path):
        """Calcola Si-SDR, SNR, PESQ, STOI, LSD confrontando i file audio."""

        # Caricamento forzato a 16kHz per compatibilità PESQ
        clean, _ = librosa.load(clean_path, sr=16000)
        deg, _ = librosa.load(degraded_path, sr=16000)
        rest, _ = librosa.load(restored_path, sr=16000)

        # Allineamento temporale (taglio alla lunghezza minima)
        min_len = min(len(clean), len(deg), len(rest))
        clean_t = torch.tensor(clean[:min_len]).unsqueeze(0).to(self.device)
        deg_t = torch.tensor(deg[:min_len]).unsqueeze(0).to(self.device)
        rest_t = torch.tensor(rest[:min_len]).unsqueeze(0).to(self.device)

        results = {}

        # 1. Si-SDR (Scale-Invariant Signal-to-Distortion Ratio)
        results['sisdr_baseline'] = scale_invariant_signal_distortion_ratio(deg_t, clean_t).item()
        results['sisdr_restored'] = scale_invariant_signal_distortion_ratio(rest_t, clean_t).item()

        # 2. SNR (Signal-to-Noise Ratio)
        results['snr_baseline'] = self._calculate_snr(clean_t, deg_t)
        results['snr_restored'] = self._calculate_snr(clean_t, rest_t)

        # 3. Metriche avanzate (PESQ/STOI)
        if self.pesq and self.stoi:
            try:
                results['pesq_baseline'] = self.pesq(deg_t, clean_t).item()
                results['pesq_restored'] = self.pesq(rest_t, clean_t).item()
                results['stoi_baseline'] = self.stoi(deg_t, clean_t).item()
                results['stoi_restored'] = self.stoi(rest_t, clean_t).item()
            except Exception as e:
                print(f"Errore calcolo PESQ/STOI: {e}")
                results['pesq_baseline'] = 0.0; results['pesq_restored'] = 0.0
                results['stoi_baseline'] = 0.0; results['stoi_restored'] = 0.0

        # 4. LSD (Log-Spectral Distance)
        results['lsd_baseline'] = self._calculate_lsd(clean_t, deg_t)
        results['lsd_restored'] = self._calculate_lsd(clean_t, rest_t)

        # Calcolo dei Delta (Miglioramenti)
        results['delta_sisdr'] = results['sisdr_restored'] - results['sisdr_baseline']
        results['delta_pesq'] = results.get('pesq_restored', 0) - results.get('pesq_baseline', 0)

        return results

    def _calculate_snr(self, target, preds):
        noise = target - preds
        signal_power = torch.mean(target ** 2)
        noise_power = torch.mean(noise ** 2)
        if noise_power < 1e-6: return 50.0 # Cap per audio identici
        return 10 * torch.log10(signal_power / noise_power).item()

    def _calculate_lsd(self, clean, estimate):
        clean_stft = torch.stft(clean.squeeze(), n_fft=2048, hop_length=512, return_complex=True)
        est_stft = torch.stft(estimate.squeeze(), n_fft=2048, hop_length=512, return_complex=True)
        clean_log = torch.log(torch.abs(clean_stft) + 1e-8)
        est_log = torch.log(torch.abs(est_stft) + 1e-8)
        diff = (clean_log - est_log) ** 2
        return torch.mean(torch.sqrt(torch.mean(diff, dim=1))).item()
