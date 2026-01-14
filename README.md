***

# 🎵 Audio-Restoration-for-Generative-Models

Le cartelle riassumono in maniera divisa e strutturata l'intero lavoro svolto.

### 📂 Modelli
Le varie versioni del modello si trovano nella cartella `models` e la versione funzionante è la **v2.2** `best_model.pt`.

***

### 💻 Codice Sorgente (`src`)
Nella cartella `src` sono presenti:

| File | Descrizione |
| :--- | :--- |
| `model.py` | Il modello stesso. |
| `data_loader.py` | Il data loader. |
| `train.py` | Il trainer. |
| `metrics.py` | Le metriche utilizzate e settate. |
| `inference.py` | Il file di inferenza. |
| `logger.py` | Un logger utile per salvare velocemente le metriche, gli spettrogrammi, commenti personali (opzionali) e le versioni Clean, Degraded e Restored di ogni audio in una cartella. |

***

### 📓 Notebooks

Nella cartella `notebook`, ci sono 3 file `.ipynb` creati su Google Colab con cui ho creato il dataset, l'ho manipolato, sistemato e dove ho scritto il modello e i file presenti in src.

> **Nota:** Ho lasciato tutte le varie versioni delle varie celle che ho creato divise per sezioni, quindi consiglio sempre di guardare la struttura delle sezioni per essere sicuri di scegliere sempre l'ultima versione (piano piano, ho sempre aggiunto qualche modifica utile).

#### 1. `modello lightweight u net.ipynb`
In particolare, il file "modello lightweight u net.ipynb" è il più importante perché permette di capire esattamente come ho lavorato e cosa ho implementato tramite tutti i commenti e la struttura delle sezioni.

*   **Per replicare gli esperimenti:** consiglio di usare la sezione *"Cella di ascolto v3: lista di canzoni da testare insieme"* dove, aggiustando i path del modello e scrivendo più nomi di canzoni insieme, vengono in automatico messe in output la versione Clean, la degradata e la Restored, le metriche e viene salvato tutto quello che il logger richiede nella cartella `results_experiments`.
*   **Per restauro diretto:** In alternativa, se si vuole solo restaurare una audio direttamente, è già pronta la cella subito dopo in cui va specificato solo il path di input e il path di output (sezione: *"Restauro di un audio diretto"*).

#### 2. `music_downloader.ipynb`
Permette di scaricare velocemente brani data una playlist di Spotify e, come se fosse una "lista della spesa", scaricherà le corrispondenti canzoni da YouTube.

#### 3. `dataset_preparation.ipynb`
Ti aiuta a creare il dataset, con la struttura Clean e Degraded richiesta (per degradare l'audio esattamente come MusicGen, usare la sezione 4: *"Simulazione compressione di MusicGen (Secondo Dataset)"*).

***

### 📊 Risultati e Dati

*   **Risultati Esperimenti:** I miei risultati sono visibili al seguente link: [Google Drive - Results](https://drive.google.com/drive/folders/1j9hIQpSb-hefFqUPkPYh21PE0eIJqML-?usp=sharing)
*   **Dataset:** I dati sono tutti presenti a questo link [Google Drive - Data](https://drive.google.com/drive/folders/1Z6h3eqPg1DKFMTR8-22eILPwiBjcuvB5?usp=sharing) e sono divisi in:
    *   `data`: quelli che ho effettivamente usato per il training, dove `musicgen_finetuning` è il secondo dataset che ha funzionato.
    *   `samples`: le canzoni originali delle 2 playlists.

### 📈 Training Graf
Nella cartella `training graf` sono presenti i grafici di confronto di wandb nelle versioni 1.1-1.2 (in cui si vede chiaramente il fenomeno di Warm Restart) e le versioni 2.1-2.2, in cui abbiamo ottenuto il modello funzionante.
