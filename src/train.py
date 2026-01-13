import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import wandb
import numpy as np

# Importiamo i nostri moduli
from src.models import RestorationLoss
from src.data_loader import get_loaders

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class ModelCheckpoint:
    def __init__(self, dirpath='models/v2', best_name='best_model.pt', second_name='second_model.pt'):
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(exist_ok=True)

        self.best_path = self.dirpath / best_name
        self.second_path = self.dirpath / second_name

        self.best_loss = float('inf')
        self.second_best_loss = float('inf')

    def __call__(self, val_loss, model):
        # Caso 1: Nuova Loss è la MIGLIORE ASSOLUTA
        if val_loss < self.best_loss:
            # Sposta il vecchio best al secondo posto (se esiste)
            if self.best_path.exists():
                # Aggiorno anche il valore della loss del secondo
                self.second_best_loss = self.best_loss
                # Per coerenza logica: quello che ERA best ora diventa second.
                import shutil
                shutil.copy2(self.best_path, self.second_path)
                print(f"   🥈 Il vecchio best diventa Second Best ({self.second_best_loss:.4f})")

            # Salva il nuovo best
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.best_path)
            print(f"   🥇 NUOVO BEST MODEL SALVATO! (Loss: {val_loss:.4f})")

        # Caso 2: Non è record assoluto, ma è meglio del secondo posto
        elif val_loss < self.second_best_loss:
            self.second_best_loss = val_loss
            torch.save(model.state_dict(), self.second_path)
            print(f"   🥈 Nuovo Second Best salvato (Loss: {val_loss:.4f})")


class AudioTrainer:
    def __init__(self, model, train_loader, val_loader, learning_rate=1e-3, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate # Salviamo per loggarlo

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate) #Versione evoluta di SGD Mini-Batch con Momentum
        self.criterion = RestorationLoss()

        #questo scheduler mi permette di abbassare il Learning Rate se la loss rimane uguale per 'patience' volte
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2,
        )

        self.early_stopping = EarlyStopping(patience=4)
        self.checkpoint = ModelCheckpoint(dirpath='models/v2') # MODIFIED: Changed 'filepath' to 'dirpath'

        # --- WANDB INIT ---
        # Inizializziamo il run qui dentro
        wandb.init(
            project="audio-restoration",
            config={
                "learning_rate": learning_rate,
                "architecture": "LightweightUNet",
                "dataset": "CustomAudio500",
                "epochs": "dynamic"
            }
        )
        # Magic: traccia automaticamente gradienti e parametri del modello
        wandb.watch(self.model, log="all", log_freq=10)

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        loop = tqdm(self.train_loader, desc="Training", leave=False)
        for degraded, clean in loop:
            degraded, clean = degraded.to(self.device), clean.to(self.device)

            output = self.model(degraded)
            loss = self.criterion(output, clean)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            #Gradient Clipping: previene l'Exploding Gradient (comune con l'audio).
            #Se la "forza" complessiva (norma L2) dei gradienti supera 1.0,
            #li scala tutti verso il basso proporzionalmente per mantenere la stabilità.

            self.optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item()) #stampa pulito il valore della loss insieme alla barra di tqdm

            # Log per step
            wandb.log({"batch_train_loss": loss.item()})

        return total_loss / len(self.train_loader)

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for degraded, clean in self.val_loader:
                degraded, clean = degraded.to(self.device), clean.to(self.device)
                output = self.model(degraded)
                loss = self.criterion(output, clean)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def fit(self, epochs):
        print(f"🚀 Avvio Training per {epochs} epoche (Logged on WandB)...")

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()

            print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            # --- WANDB LOGGING ---
            # Logghiamo le metriche principali alla fine di ogni epoca
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            })

            self.scheduler.step(val_loss)
            self.checkpoint(val_loss, self.model)
            self.early_stopping(val_loss)

            if self.early_stopping.early_stop:
                print("🛑 Early Stopping attivato!")
                break

        print("🎉 Training completato.")
        wandb.finish() # Chiude il run correttamente
