# Tutorial: Rulare pe FEP cu Apptainer + Ollama

Ghid complet pas cu pas pentru rularea pipeline-ului de ontologii pe clusterul GPU al facultății (FEP / UPB Grid), folosind Apptainer pentru Ollama.

---

## Cuprins

1. [Cum funcționează FEP + Apptainer](#1-cum-functioneaza-fep--apptainer)
2. [Conectare la FEP](#2-conectare-la-fep)
3. [Setup inițial pe FEP (o singură dată)](#3-setup-initial-pe-fep-o-singura-data)
   - 3.1 [Apptainer + Ollama SIF](#31-apptainer--ollama-sif)
   - 3.2 [Copiere proiect](#32-copiere-proiect)
   - 3.3 [Python venv + dependențe](#33-python-venv--dependente)
   - 3.4 [Descărcare model LLM](#34-descarcare-model-llm)
   - 3.5 [Configurare .env](#35-configurare-env)
4. [Rulare o singură dată](#4-rulare-o-singura-data)
5. [Rulări multiple: paralel sau secvențial](#5-rulari-multiple-paralel-sau-secvential)
6. [Monitorizare joburi](#6-monitorizare-joburi)
7. [Extragere rezultate pe laptop](#7-extragere-rezultate-pe-laptop)
8. [Referință rapidă: Flags disponibili](#8-referinta-rapida-flags-disponibili)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Cum funcționează FEP + Apptainer

```
Tu (laptop)  ──SSH──►  Login node (fep.grid.pub.ro)
                              │
                              │  ./fep/launch.sh -n 5
                              │     → sbatch fep/job.sh
                              ▼
                        SLURM Scheduler
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              GPU Node 1          GPU Node 2
         ┌──────────────────┐  ┌──────────────────┐
         │ apptainer run    │  │ apptainer run    │
         │  ollama.sif serve│  │  ollama.sif serve│
         │  (port 11434)    │  │  (port 11435)    │
         │ python src/main  │  │ python src/main  │
         └──────────────────┘  └──────────────────┘
```

**Principii cheie:**
- **Nu rula cod greu pe login node** — acela e pentru comenzi SLURM și setup
- `ollama.sif` este imaginea containerului Apptainer care conține Ollama
- Modelele sunt stocate în `~/ollama_project/ollama_storage/` — descărcate o dată, reutilizate la infinit
- `--nv` în Apptainer = accesul containerului la GPU-ul NVIDIA alocat de SLURM
- Fiecare rulare paralelă are port Ollama unic (`11434 + task_id`) → fără conflicte
- Poți **închide laptopul** — jobul rulează independent pe cluster

---

## 2. Conectare la FEP

```bash
ssh stefan.covaliu@fep.grid.pub.ro
```

**Opțional — cheie SSH (fără parolă la fiecare conectare):**

```bash
# Pe laptop — generează cheie dacă nu ai
ssh-keygen -t ed25519 -C "fep-grid"

# Copiaz-o pe FEP
ssh-copy-id stefan.covaliu@fep.grid.pub.ro
```

---

## 3. Setup inițial pe FEP (o singură dată)

### 3.1 Apptainer + Ollama SIF

Ai deja fcut asta, dar îl documentăm pentru referință:

```bash
# Conectează-te la FEP
ssh stefan.covaliu@fep.grid.pub.ro

# Creează directorul de lucru Ollama
mkdir -p ~/ollama_project && cd ~/ollama_project

# Creează directorul unde vor fi stocate modelele
mkdir -p ./ollama_storage

# Descarcă imaginea Ollama ca Apptainer SIF (~1.5 GB)
apptainer pull ollama.sif docker://ollama/ollama:latest
```

> Dacă `apptainer` nu e în PATH, încearcă: `module load apptainer` sau `module load singularity`

**Verifică că SIF-ul există:**

```bash
ls -lh ~/ollama_project/ollama.sif
# Ar trebui să afișeze ceva de ~1.5-2 GB
```

### 3.2 Copiere proiect

**Pe laptop** — copiază proiectul pe FEP:

```bash
# Sincronizare completă (rulează de pe laptop)
rsync -avz \
    --exclude='venv/' \
    --exclude='__pycache__/' \
    --exclude='data/runs/' \
    --exclude='data/parallel/' \
    --exclude='data/output/' \
    ~/onto4/ stefan.covaliu@fep.grid.pub.ro:~/covaliu/
```

> Poți re-rula oricând pentru a sincroniza modificări noi de cod.

**Dacă nu ai `rsync`** (Windows fără WSL):

```bash
scp -r ~/onto4 stefan.covaliu@fep.grid.pub.ro:~/covaliu
```

### 3.3 Python venv + dependențe

**Pe FEP** (după ce ai copiat proiectul):

```bash
cd ~/covaliu

# Verifică Python disponibil (trebuie >= 3.9)
python3 --version

# Dacă nu e disponibil, caută modulul:
# module avail python
# module load python/3.11

# Creează virtual environment
python3 -m venv venv

# Activează
source venv/bin/activate

# Instalează dependențele
pip install --upgrade pip
pip install -r requirements.txt

# Verifică
python -c "from langchain_ollama import ChatOllama; print('OK — langchain_ollama disponibil')"
```

> Venv-ul se creează o singură dată. Joburile SLURM îl activează automat.

### 3.4 Descărcare model LLM

Modelele se descarcă de pe **login node** (are acces la internet) și se stochează în `~/ollama_project/ollama_storage/`. Descărcarea se face o singură dată — după aceea toate joburile le folosesc din cache.

```bash
# Pornește Ollama temporar în container (fără GPU, doar pentru download)
OLLAMA_HOST=127.0.0.1:11434 \
apptainer run \
    --bind ~/ollama_project/ollama_storage:/root/.ollama \
    ~/ollama_project/ollama.sif serve &

# Așteaptă câteva secunde să pornească
sleep 5

# Descarcă modelul dorit (alege unul din variante)
OLLAMA_HOST=127.0.0.1:11434 \
apptainer exec \
    --bind ~/ollama_project/ollama_storage:/root/.ollama \
    ~/ollama_project/ollama.sif \
    ollama pull qwen2.5:14b
# Alte opțiuni:
# ollama pull qwen2.5:14b    # ~9 GB VRAM — calitate mai bună
# ollama pull llama3.1:8b    # ~5 GB VRAM — alternativă
# ollama pull mistral:7b     # ~4 GB VRAM — alternativă rapidă

# Oprește Ollama temporar
kill %1
```

**Verifică că modelul s-a descărcat:**

```bash
OLLAMA_HOST=127.0.0.1:11434 \
apptainer run \
    --bind ~/ollama_project/ollama_storage:/root/.ollama \
    ~/ollama_project/ollama.sif serve &
sleep 5

OLLAMA_HOST=127.0.0.1:11434 \
apptainer exec \
    --bind ~/ollama_project/ollama_storage:/root/.ollama \
    ~/ollama_project/ollama.sif \
    ollama list
# Ar trebui să apară: qwen2.5:7b   ...   4.7 GB

kill %1
```

### 3.5 Configurare .env

```bash
cd ~/covaliu

# Copiază configurația pentru FEP (fără API keys)
cp .env.fep .env

# Verifică că e corect
cat .env
# Trebuie să conțină: LLM_BACKEND=ollama și API keys goale
```

Fișierul `.env.fep` e deja pregătit:
- `LLM_BACKEND=ollama` — fără API keys
- `RATE_LIMIT_RPM=9999` — fără rate limiting
- `GOOGLE_API_KEY=`, `OPENAI_API_KEY=`, `HUGGINGFACE_API_KEY=` — goale

**Verificare rapidă că proiectul e configurat corect:**

```bash
source venv/bin/activate
python -c "from config.settings import settings; print('Backend:', settings.LLM_BACKEND, '| Model:', settings.OLLAMA_MODEL)"
# Trebuie să afișeze: Backend: ollama | Model: qwen2.5:7b
```

---

## 4. Moduri de Rulare: Comenzi și Explicații

Mai jos sunt toate modurile prin care poți lansa proiectul folosind scriptul `./fep/launch.sh`. Asigură-te că ești în directorul `~/covaliu` pe FEP.

### Alegerea Partiției SLURM
Înainte de a rula, poți alege ce fel de GPU dorești adăugând `--partition <nume>`. În mod implicit este setată partiția `dgxa100` (nodurile puternice).
- `--partition dgxa100` (sau `gpu` în funcție de configurația FEP)
Exemplu: `./fep/launch.sh --input "sample.pdf" --partition dgxa100`

### 4.1 Rulare Standard (un singur job)
Lansează o singură execuție. Este modul de bază pentru un document normal (rezultatele apar în `data/fep/job_<ID>/run_0`).
```bash
./fep/launch.sh --input "sample.pdf"
# Sau selectând explicit partiția:
./fep/launch.sh --input "sample.pdf" --partition dgxa100
```

### 4.2 Documente Mari (Job Chaining)
Folosit când un document e prea mare și s-ar opri din cauza limitei de timp SLURM. Lansează `N` job-uri înlănțuite. Primul rulează, restul așteaptă. Când primul se termină/expiră, următorul preia cache-ul și continuă exact de unde a rămas, în **același folder**.
```bash
./fep/launch.sh --input "document_urias.pdf" --chain 5 --partition dgxa100
```

### 4.3 Reluare la Eșec (Resume)
Dacă un job anterior a crăpat, a expirat (Timeout) sau pur și simplu vrei să continui procesarea pe baza unui cache parțial existent, folosești `--resume` dându-i calea către folderul job-ului inițial.
```bash
./fep/launch.sh --input "document_ramas.pdf" --resume "data/fep/job_134567" --partition dgxa100
```

### 4.4 Rulări Multiple: Paralelism / Secvențial (Teste)
Acest mod îl folosești pentru a rula **același input de mai multe ori** (pentru a testa halucinațiile LLM-ului în mai multe run-uri distincte). 

**Recomandat (Pe același nod de GPU pentru a nu umple coada SLURM):**
```bash
# Rulează de 3 ori SIMULTAN pe același job/GPU alocat
./fep/launch.sh --input "sample.pdf" -n 3 --parallel --tasks single --partition dgxa100

# Rulează de 3 ori SECVENȚIAL (pe rând, pentru a evita panica Out Of Memory pe model mare, dar pe același job)
./fep/launch.sh --input "sample.pdf" -n 3 --sequential --tasks single --partition dgxa100
```

**Separat (Cere multiple GPU-uri distincte de la cluster):**
```bash
# Cere 3 job-uri separate (3 GPU-uri diferite) - folosind array SLURM
./fep/launch.sh --input "sample.pdf" -n 3 --parallel --tasks separate --partition dgxa100
```

---

**Modul Array (GPU-uri Separate)**
Asta va plasa fiecare "Run" pe rândul propriu de pe Cluster.
```bash
./fep/launch.sh -n 2 --parallel --tasks separate
```

### Structura directoarelor după rulare

```
~/covaliu/data/fep/
├── job_34215243/               ← ID-ul asignat de SLURM (sau pentru Chaining, baza)
│   ├── run_0/                  ← prima rulare din acest job
│   │   ├── cache/              ← cache MD5 izolat (salvarea stadiului)
│   │   ├── chunks/             ← document procesat în chunk-uri
│   │   ├── output/             ← fișiere finale ontologice
│   │   └── runs/               ← arhive zilnice
│   ├── run_1/                  ← (Dacă folosești -n > 1) a doua rulare 
│   └── run_2/                  ← (Dacă folosești -n > 1) a treia rulare 
└── job_34215244/               ← Job Chain 2 (continuare logică care refolosește datele)
```

---

## 6. Monitorizare joburi

```bash
# Lista joburilor tale
squeue -u $USER

# Live view (actualizare la fiecare 5 secunde)
watch -n 5 squeue -u $USER

# Informații detaliate despre un job
scontrol show job <JOB_ID>

# Anulează un job specific
scancel <JOB_ID>

# Anulează toate joburile tale
scancel -u $USER

# Urmărire live a log-ului unui job (înlocuiește cu ID-ul tău)
tail -f fep/logs/<JOB_ID>_0.out

# Urmărire simultană a tuturor log-urilor active
tail -f fep/logs/*.out

# Verifică statusul după finalizare
tail -30 fep/logs/<JOB_ID>_0.out
# Caută liniile: STATUS: SUCCESS sau STATUS: FAILED

# Statistici detaliate (timp, memorie, exit code)
sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

**Statusuri SLURM:**

| Status | Semnificație |
|--------|-------------|
| `PD`   | Pending — așteaptă GPU liber |
| `R`    | Running — rulează acum |
| `CG`   | Completing — se finalizează |
| `CD`   | Completed — terminat cu succes |
| `F`    | Failed — eroare, verifică log-ul `.err` |
| `TO`   | Timeout — a depășit limita de timp |

---

## 7. Extragere rezultate pe laptop

Rulează pe **laptop** (nu pe FEP):

### Copiază toate rezultatele

```bash
rsync -avz \
    stefan.covaliu@fep.grid.pub.ro:~/covaliu/data/fep/ \
    ~/onto4/data/fep_results/
```

### Copiază doar fișierele finale (ontologii, fără cache)

```bash
rsync -avz \
    --include='*/' \
    --include='*.json' \
    --include='*.ttl' \
    --exclude='*' \
    stefan.covaliu@fep.grid.pub.ro:~/covaliu/data/fep/ \
    ~/onto4/data/fep_results/
```

### Copiază log-urile

```bash
rsync -avz \
    stefan.covaliu@fep.grid.pub.ro:~/covaliu/fep/logs/ \
    ~/covaliu/fep/logs_fep/
```

### Raport comparativ (după ce ai adus rezultatele)

```bash
cd ~/covaliu
source venv/bin/activate
python src/final_report.py
```

---

## 8. Referință rapidă: Flags disponibili

### `./fep/launch.sh`

| Flag | Descriere | Default |
|------|-----------|---------|
| `-n N` | Număr de rulări | `1` |
| `--tasks` | Organizare taskuri (`single` sau `separate`) | `single` |
| `--sequential` | Rulare secvențială (în mod single sau separate) | paralel |
| `--model MODEL` | Model Ollama | `.env` |
| `--no-cache` | Dezactivează cache LLM | cache activ |
| `--fresh` | Șterge cache înainte de rulare | - |
| `--blueprint-only` | Doar faza 1+2 (schema) | full pipeline |
| `--data-dir DIR` | Director bază pentru rulări | `data/fep` |
| `--time HH:MM:SS` | Limita de timp SLURM | `06:00:00` |
| `--partition PART` | Partiție SLURM | `gpu` |
| `--project-dir DIR` | Calea proiectului | `~/covaliu` |
| `--sif PATH` | Calea la `ollama.sif` | `~/ollama_project/ollama.sif` |
| `--storage DIR` | Directorul de modele Ollama | `~/ollama_project/ollama_storage` |

### `python src/main.py` (direct)

| Flag | Descriere |
|------|-----------|
| `--data-dir DIR` | Director izolat pentru această rulare |
| `--no-cache` | Dezactivează cache MD5 |
| `--fresh` | Șterge cache și reface toate apelurile LLM |
| `--blueprint-only` | Doar schema (T-Box), fără populare KG |
| `--reset-rate-limit` | Resetează contorul de rate limiting |

---

## 9. Troubleshooting

### `apptainer: command not found`

```bash
# Caută și încarcă modulul
module avail apptainer
module avail singularity

# Încearcă
module load apptainer
# sau
module load singularity

# Adaugă în ~/.bashrc ca să fie automat la fiecare login
echo 'module load apptainer' >> ~/.bashrc
```

### Jobul e în pending mult timp (`PD`)

```bash
# Verifică motivul
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.8T %.10M %.9l %.6D %R"

# Listează partițiile disponibile și starea lor
sinfo -s
```

Motive comune:
- `Resources` — nu sunt GPU-uri libere, așteptare normală
- `Priority` — alte joburi au prioritate mai mare
- `QOSMaxJobsPerUser` — ai atins limita de joburi simultane, anulează câteva cu `scancel`
- Partiția `gpu` nu există — verifică cu `sinfo -s` și folosește `--partition <alta>`

### Eroare: `Ollama SIF nu găsit`

```bash
ls -lh ~/ollama_project/ollama.sif

# Dacă nu există, descarcă din nou
cd ~/ollama_project
apptainer pull ollama.sif docker://ollama/ollama:latest
```

### Eroare: `venv not found`

```bash
cd ~/covaliu
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Jobul pică cu `CUDA out of memory`

Modelul e prea mare pentru VRAM-ul GPU-ului. Soluții:

```bash
# Folosește model mai mic (4.7 GB VRAM)
./fep/launch.sh -n 3 --model qwen2.5:7b

# Sau cere mai multă memorie GPU
# Editează fep/job.sh: #SBATCH --mem=32G
# (dacă FEP permite, verifică cu scontrol show partition gpu)
```

### Ollama pornește dar modelul nu răspunde / nu e descărcat

```bash
# Pe login node, verifică modelele disponibile
OLLAMA_HOST=127.0.0.1:11434 \
apptainer run \
    --bind ~/ollama_project/ollama_storage:/root/.ollama \
    ~/ollama_project/ollama.sif serve &
sleep 5

OLLAMA_HOST=127.0.0.1:11434 \
apptainer exec \
    --bind ~/ollama_project/ollama_storage:/root/.ollama \
    ~/ollama_project/ollama.sif \
    ollama list

# Dacă modelul nu apare, descarcă-l
OLLAMA_HOST=127.0.0.1:11434 \
apptainer exec \
    --bind ~/ollama_project/ollama_storage:/root/.ollama \
    ~/ollama_project/ollama.sif \
    ollama pull qwen2.5:7b

kill %1
```

### Ollama nu pornește în 90s în job

Posibil timeout prea scurt la pornirea containerului. Verifică log-ul:

```bash
cat fep/logs/<JOB_ID>_<TASK_ID>.err
```

Dacă e eroare de GPU, încearcă să te asiguri că `--gres=gpu:1` e în SBATCH.

### Rulările paralele interferează

Fiecare rulare folosește port unic (`11434 + TASK_ID`) și director de date separat (`data/parallel/run_N`). Dacă totuși apare conflict, verifică:

```bash
grep "OLLAMA_PORT\|DATA_DIR\|TASK_ID" fep/job.sh
```

---

## Flux complet — De la zero la rezultate

Pași care trebuie făcuți **o singură dată** (setup inițial):

```bash
# 1. Conectare
ssh stefan.covaliu@fep.grid.pub.ro

# 2. Apptainer + Ollama (ai făcut deja asta)
mkdir -p ~/ollama_project/ollama_storage
# apptainer pull ollama.sif docker://ollama/ollama:latest  ← deja făcut

# 3. Copiere proiect (de pe laptop)
rsync -avz --exclude='venv/' --exclude='data/runs/' --exclude='data/parallel/' \
    ~/covaliu/ stefan.covaliu@fep.grid.pub.ro:~/covaliu/

# 4. Setup Python (pe FEP)
cd ~/covaliu
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 5. Descărcare model (pe FEP — login node)
OLLAMA_HOST=127.0.0.1:11434 apptainer run \
    --bind ~/ollama_project/ollama_storage:/root/.ollama \
    ~/ollama_project/ollama.sif serve &
sleep 5
OLLAMA_HOST=127.0.0.1:11434 apptainer exec \
    --bind ~/ollama_project/ollama_storage:/root/.ollama \
    ~/ollama_project/ollama.sif ollama pull qwen2.5:7b
kill %1

# 6. Configurare .env
cp .env.fep .env

# 7. Permisiuni scripturi
chmod +x fep/launch.sh fep/job.sh
```

Pași pentru **fiecare rulare nouă**:

```bash
# Pe FEP — lansează rulările
cd ~/covaliu
./fep/launch.sh -n 5                    # 5 rulări paralele

# Monitorizare (opțional — poți închide laptopul)
watch -n 10 squeue -u $USER

# Pe laptop — colectare rezultate după terminare
rsync -avz stefan.covaliu@fep.grid.pub.ro:~/covaliu/data/parallel/ \
    ~/covaliu/data/fep_results/

# Raport comparativ
cd ~/covaliu && python src/final_report.py
```
