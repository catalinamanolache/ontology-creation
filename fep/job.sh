#!/bin/bash
# ============================================================
# SLURM Job Script — Ontology Pipeline pe FEP cu Apptainer
# ============================================================
# Ollama rulează în containerul ~/ollama_project/ollama.sif
# Modelele sunt persistate în ~/ollama_project/ollama_storage/
#
# Folosire:
#   sbatch fep/job.sh                     # 1 rulare
#   sbatch --array=0-4 fep/job.sh         # 5 rulări paralele
#
# De obicei apelat prin fep/launch.sh — vezi TUTORIAL_FEP.md
# ============================================================

#SBATCH --job-name=onto-pipeline
#SBATCH --partition=gpu              # Partiție GPU — verifică cu: sinfo -s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1                 # 1 GPU per job
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=fep/logs/%A_%a.out  # %A=job id, %a=array task id
#SBATCH --error=fep/logs/%A_%a.err

# ── Identifică task-ul curent ────────────────────────────────
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
JOB_ID=${SLURM_JOB_ID:-local}

echo "============================================================"
echo "  AbA Ontology Pipeline — FEP Job (Apptainer)"
echo "  Job ID    : ${JOB_ID}"
echo "  Task ID   : ${TASK_ID}"
echo "  Node      : $(hostname)"
echo "  Date      : $(date)"
echo "============================================================"

# ── Directorul proiectului ───────────────────────────────────
PROJECT_DIR="${PROJECT_DIR:-$HOME/covaliu}"
cd "$PROJECT_DIR" || { echo "ERROR: Cannot cd to $PROJECT_DIR"; exit 1; }

# ── Apptainer — calea SIF și storage modele ──────────────────
OLLAMA_SIF="${OLLAMA_SIF:-$HOME/ollama_project/ollama.sif}"
OLLAMA_STORAGE="${OLLAMA_STORAGE:-$HOME/ollama_project/ollama_storage}"

if [ ! -f "$OLLAMA_SIF" ]; then
    echo "ERROR: Ollama SIF nu găsit la $OLLAMA_SIF"
    echo "       Rulează: cd ~/ollama_project && apptainer pull ollama.sif docker://ollama/ollama:latest"
    exit 1
fi
mkdir -p "$OLLAMA_STORAGE"
echo "[OK] Ollama SIF: $OLLAMA_SIF"
echo "[OK] Modele storage: $OLLAMA_STORAGE"

# ── Python virtual environment ───────────────────────────────
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv}"
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: venv nu a fost găsit la $VENV_DIR"
    echo "       Rulează: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi
source "$VENV_DIR/bin/activate"
echo "[OK] Python venv activat: $(python --version)"

# ── Director de date izolat pentru această rulare ────────────
if [ -n "${RESUME_DIR:-}" ]; then
    # Overriden base directory for chaining/resuming
    DATA_DIR="$RESUME_DIR"
elif [ "${TASKS_MODE:-separate}" != "single" ]; then
    DATA_DIR="$PROJECT_DIR/data/fep/job_${JOB_ID}_task_${TASK_ID}"
else
    # In modul single-task the exact dir is computed further down.
    DATA_DIR="$PROJECT_DIR/data/fep/job_${JOB_ID}"
fi
mkdir -p "$DATA_DIR"
echo "[OK] Director date bază: $DATA_DIR"

# ── Port unic Ollama per task ─────────────────────────────────
# Evită conflicte dacă mai multe joburi rulează pe același nod
OLLAMA_PORT=$((11434 + TASK_ID))
echo "[OK] Port Ollama: $OLLAMA_PORT"

# ── Pornire server Ollama în container ───────────────────────
# --nv = acces la NVIDIA GPU
# --bind = montează directorul de modele în container
# OLLAMA_HOST = adresa pe care serverul ascultă în interiorul containerului
echo "[*] Pornesc Ollama (Apptainer) pe portul ${OLLAMA_PORT}..."

export APPTAINERENV_OLLAMA_HOST="0.0.0.0:${OLLAMA_PORT}"
apptainer run --nv \
    --bind "${OLLAMA_STORAGE}:/root/.ollama" \
    "${OLLAMA_SIF}" serve > "$PROJECT_DIR/fep/logs/ollama_${JOB_ID}_${TASK_ID}.log" 2>&1 &

OLLAMA_PID=$!
echo "[OK] Ollama PID: $OLLAMA_PID"

# Așteaptă până când serverul e gata (max 90 secunde)
READY=0
for i in $(seq 1 45); do
    if curl -sf "http://127.0.0.1:${OLLAMA_PORT}/api/tags" > /dev/null 2>&1; then
        READY=1
        echo "[OK] Ollama gata după $((i*2))s"
        break
    fi
    sleep 2
done

if [ "$READY" -eq 0 ]; then
    echo "ERROR: Ollama nu a pornit în 90s. Abandon."
    kill "$OLLAMA_PID" 2>/dev/null
    exit 1
fi

# ── Extragere model ─────────
if [ -z "$OLLAMA_MODEL" ] && [ -f "$PROJECT_DIR/.env" ]; then
    export $(grep '^OLLAMA_MODEL=' "$PROJECT_DIR/.env" | xargs)
fi
MODEL="${OLLAMA_MODEL}"
if [ -z "$MODEL" ]; then
    echo "ERROR: Model nespecificat (verifica flag launch, settings sau .env)"
    kill "$OLLAMA_PID" 2>/dev/null
    exit 1
fi

# ── Pull model dacă nu e deja în cache ───────────────────────
echo "[*] Verificare model in local storage ($MODEL)..."
export APPTAINERENV_OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
if apptainer exec --nv --bind "${OLLAMA_STORAGE}:/root/.ollama" "${OLLAMA_SIF}" ollama list | grep -q "$MODEL"; then
    echo "[OK] Model deja prezent in cache: $MODEL"
else
    echo "[*] Descarcare model: $MODEL (Asteapta, dureaza...)"
    apptainer exec --nv \
        --bind "${OLLAMA_STORAGE}:/root/.ollama" \
        "${OLLAMA_SIF}" ollama pull "$MODEL"
    echo "[OK] Model descarcat cu succes: $MODEL"
fi

# ── Rulare pipeline ──────────────────────────────────────────
export DATA_DIR
export OLLAMA_BASE_URL="http://127.0.0.1:${OLLAMA_PORT}"
export LLM_BACKEND=ollama
export OLLAMA_MODEL="$MODEL"
export RATE_LIMIT_RPM=9999
export RATE_LIMIT_RPD=999999
export MIN_DELAY=0.0

echo "[*] Pornesc pipeline cu argumente: ${EXTRA_ARGS:-<niciunul>}"

EXTRA_PY_ARGS=""
if [ -n "${INPUT_FILE:-}" ]; then
    EXTRA_PY_ARGS="--input $INPUT_FILE"
    echo "  [Input File] $INPUT_FILE"
fi

echo "------------------------------------------------------------"

if [ "${TASKS_MODE:-separate}" = "single" ]; then
    echo "[*] Mod SINGLE TASK (Acelasi job/GPU) — Iteratii: ${N_RUNS:-1} | Stil: ${RUN_MODE:-parallel}"
    pids=()
    PIPELINE_EXIT=0
    
    for i in $(seq 0 $((${N_RUNS:-1} - 1))); do
        # Chaining/Resuming in single task mode normally runs 1 iteration. 
        # Make sure to strictly bind to a single run folder if we are resuming to prevent it from dropping data in the parent folder.
        R_DIR="$DATA_DIR/run_${i}"
        mkdir -p "$R_DIR"
        
        if [ "${N_RUNS:-1}" -eq 1 ]; then
            echo "  [Start] Executie in logul principal -> $R_DIR"
            DATA_DIR="$R_DIR" python -u src/main.py ${EXTRA_ARGS:-} $EXTRA_PY_ARGS --data-dir "$R_DIR"
            PIPELINE_EXIT=$?
        elif [ "${RUN_MODE}" = "parallel" ]; then
            echo "  [Start Paralel] Iteratia $i -> Output: fep/logs/${JOB_ID}_run_${i}.out"
            DATA_DIR="$R_DIR" python -u src/main.py ${EXTRA_ARGS:-} $EXTRA_PY_ARGS --data-dir "$R_DIR" > "fep/logs/${JOB_ID}_run_${i}.out" 2>&1 &
            pids+=($!)
        else
            echo "  [Start Secvential] Iteratia $i -> $R_DIR"
            DATA_DIR="$R_DIR" python -u src/main.py ${EXTRA_ARGS:-} $EXTRA_PY_ARGS --data-dir "$R_DIR" | tee "fep/logs/${JOB_ID}_run_${i}.out"
            EX=${PIPESTATUS[0]} # get exit code from python, not tee
            if [ $EX -ne 0 ]; then PIPELINE_EXIT=$EX; fi
        fi
    done
    
    if [ "${RUN_MODE}" = "parallel" ]; then
        echo "[*] Astept ca toate rularile paralele din background sa se incheie..."
        for pid in "${pids[@]}"; do
            wait "$pid"
            EX=$?
            if [ $EX -ne 0 ]; then PIPELINE_EXIT=$EX; fi
        done
        echo "[OK] Toate rularile paralele interne au luat sfarsit."
    fi
else
    echo "[*] Mod SEPARATE TASK — Executie normala in task dedicat (Array)."
    echo "    $DATA_DIR"
    python -u src/main.py ${EXTRA_ARGS:-} $EXTRA_PY_ARGS --data-dir "$DATA_DIR"
    PIPELINE_EXIT=$?
fi

echo "------------------------------------------------------------"

# ── Oprire Ollama ─────────────────────────────────────────────
echo "[*] Opresc Ollama (PID $OLLAMA_PID)..."
kill "$OLLAMA_PID" 2>/dev/null
wait "$OLLAMA_PID" 2>/dev/null

# ── Raport final ──────────────────────────────────────────────
echo ""
echo "============================================================"
if [ "$PIPELINE_EXIT" -eq 0 ]; then
    echo "  STATUS: SUCCESS"
    echo "  Rezultate: $DATA_DIR"
    echo "  Arhive:    $DATA_DIR/runs/"
else
    echo "  STATUS: FAILED (exit code $PIPELINE_EXIT)"
    echo "  Logs:  fep/logs/${JOB_ID}_${TASK_ID}.err"
fi
echo "  Terminat: $(date)"
echo "============================================================"

exit "$PIPELINE_EXIT"
