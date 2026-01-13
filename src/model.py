import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightUNet(nn.Module):
    """
    Ultra-lightweight U-Net per audio restoration.

    Architettura:
    - Encoder: 3 blocchi di convoluzione che riducono la dimensione temporale.
    - Decoder: 3 blocchi che riportano l'audio alla dimensione originale.
    - Skip Connections: Uniscono le feature dell'encoder col decoder per preservare i dettagli.

    Input: [batch, 1, 48000] (Tensore audio mono: 3 sec @ 16kHz)
    Output: [batch, 1, 48000] (Audio restaurato)
    """
    def __init__(self):
        super().__init__()

        # --- ENCODER (Contracting Path) ---
        # Aumentiamo i canali (profondità) e riduciamo il tempo (pooling)

        #Layers della rete neurale, estrazione di features:
        # All'inizio (enc1 - 16 canali): La rete guarda pezzettini piccolissimi di audio. Qui impara feature molto "basse" e grezze, tipo: c'è un picco di volume qui? C'è un silenzio lì?
        # A metà (enc2 - 32 canali): Mettendo insieme le informazioni grezze, inizia a riconoscere pattern più complessi, tipo: questa è una frequenza bassa, questo è un attacco di tamburo.
        # Verso la fine (enc3 - 64 canali): Le feature diventano ancora più astratte. La rete non vede più solo numeri, ma "concetti" sonori.

        # Blocco 1: Da 1 canale (audio mono) a 16 features
        self.enc1 = self._conv_block(1, 16)
        self.pool1 = nn.MaxPool1d(kernel_size=2) # Dimezza la lunghezza temporale (48k -> 24k)

        # Blocco 2: Da 16 a 32 features
        self.enc2 = self._conv_block(16, 32)
        self.pool2 = nn.MaxPool1d(kernel_size=2) # 24k -> 12k

        # Blocco 3: Da 32 a 64 features
        self.enc3 = self._conv_block(32, 64)
        self.pool3 = nn.MaxPool1d(kernel_size=2) # 12k -> 6k

        # --- BOTTLENECK ---
        # Il punto più "profondo" della rete. Qui l'audio è molto compresso
        # rappresentando caratteristiche astratte di alto livello in uno spazio latente.
        self.bottleneck = self._conv_block(64, 128)

        #Cosa sta realmente succedendo?
        #La rete è come se avesse seguito queste istruzioni:
        # Prendi questo audio rumoroso, estrai le sue caratteristiche fondamentali (Encoder),
        # riassumilo all'osso nel Bottleneck (così il rumore si perde per strada),
        # e poi usa quel riassunto pulito per ricostruire l'audio originale (Decoder, fase successiva)

        # --- DECODER (Expanding Path) ---
        # Usiamo ConvTranspose1d per fare l'operazione inversa (Upsampling)

        # Upsample 1: 6k -> 12k
        self.upconv3 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        # Nota: input 128 perché riceve: 64 dallo skip connection + 64 dall'upconv
        self.dec3 = self._conv_block(128, 64)

        # Dal basso (Upconv): Arriva l'informazione dal Bottleneck. È ricca di significato ("qui c'è un basso") ma povera di dettaglio spaziale.
        # Il layer upconv3 trasforma i 128 canali del bottleneck in 64 canali.
        # Dal lato (Skip Connection): La rete va a ripescare quello che aveva salvato nell'Encoder allo stesso livello (e3).
        # Quell'informazione non è mai passata per il bottleneck! È "fresca", piena di dettagli originali. Anche lei ha 64 canali.

        # Upsample 2: 12k -> 24k
        self.upconv2 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(64, 32)

        # Upsample 3: 24k -> 48k
        self.upconv1 = nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(32, 16)

        # --- OUTPUT LAYER ---
        # Comprime le 16 feature finali in 1 singolo canale audio (waveform)
        self.final = nn.Conv1d(16, 1, kernel_size=3, padding=1)

    def _conv_block(self, in_ch, out_ch):
        """
        Helper function che crea un blocco standard:
        Conv1d -> BatchNorm -> ReLU -> Conv1d -> BatchNorm -> ReLU
        """
        return nn.Sequential(
            # Prima convoluzione: estrae features
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            # Normalizzazione: stabilizza il training e lo rende più veloce
            nn.BatchNorm1d(out_ch),
            # Attivazione: introduce non-linearità
            nn.ReLU(inplace=True),

            # Seconda convoluzione (raffinamento)
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 1. Passaggio attraverso l'Encoder
        # Salviamo l'output di ogni blocco (e1, e2, e3) per le skip connections
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # 2. Passaggio nel Bottleneck
        b = self.bottleneck(p3)

        # 3. Passaggio nel Decoder con Skip Connections

        # Livello 3
        u3 = self.upconv3(b)
        # Fix dimensioni: A volte il pooling arrotonda per difetto.
        # Se c'è discrepanza di 1 sample tra encoder e decoder, interpoliamo.
        if u3.shape != e3.shape:
            u3 = F.interpolate(u3, size=e3.shape[2:])
        #     forza u3 ad avere la stessa dimensione temporale di e3 (cioè lo stesso numero di campioni/posizioni lungo l’asse “tempo”).
        #     In pratica è un resize 1D: se u3 è lungo 6001 e e3 è lungo 6000, l’interpolazione lo riporta a 6000 (o viceversa),
        #     “stirando” o “comprimendo” leggermente i valori lungo il tempo.
        # Concatenazione: uniamo l'audio ricostruito (u3) con i dettagli originali (e3)
        u3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(u3)

        # Livello 2
        u2 = self.upconv2(d3)
        if u2.shape != e2.shape:
            u2 = F.interpolate(u2, size=e2.shape[2:])
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)

        # Livello 1
        u1 = self.upconv1(d2)
        if u1.shape != e1.shape:
            u1 = F.interpolate(u1, size=e1.shape[2:])
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)

        # 4. Output finale
        out = self.final(d1)

        # RESIDUAL CONNECTION:
        # Invece di imparare l'audio da zero, la rete impara solo il "rumore" da sottrarre.
        # Out = Input (sporco) + Correzione
        out = out + x

        return out

class RestorationLoss(nn.Module):
    """
    Loss Function Ibrida:
    Combina l'errore nel tempo (forma dell'onda) con l'errore in frequenza (spettro).
    """
    def __init__(self):
        super().__init__()

    def forward(self, predicted, target):
        # 1. Loss nel dominio del tempo (L1 Loss = Mean Absolute Error)
        # Controlla se l'onda sonora combacia punto per punto
        loss_l1 = F.l1_loss(predicted, target)

        # 2. Loss nel dominio delle frequenze (FFT)
        # Trasformiamo l'audio in frequenze per controllare se il tono/timbro è giusto.
        # torch.fft.rfft calcola la Fast Fourier Transform per input reali (trasformata di Fourier).
        pred_fft = torch.fft.rfft(predicted, dim=-1)
        targ_fft = torch.fft.rfft(target, dim=-1)

        # Prendiamo solo la magnitudine (abs) ignorando la fase per ora
        # la fase è molto difficile da impare per una rete neurale,
        # inoltre l'orecchio umano percepisce meglio i cambi di magnitudine che di fase
        pred_mag = torch.abs(pred_fft)
        targ_mag = torch.abs(targ_fft)

        loss_freq = F.l1_loss(pred_mag, targ_mag)

        # Combiniamo le due loss. Il peso 0.5 è empirico.
        return loss_l1 + 0.5 * loss_freq
