#!/bin/bash
# ============================================================
# FEP Launch Script — Submite una sau mai multe rulări
# ============================================================
# Folosire:
#   ./fep/launch.sh [opțiuni]
#
# Opțiuni:
#   -n, --runs N           Număr de rulări (default: 1)
#   --sequential           Rulare secvențială internă sau externă
#   --parallel             Rulare paralelă (implicit).
#   --tasks single|separate Daca rularea (secvential/paralel) are loc intern, 
#                          pe acelasi job/task (single), sau prin array SLURM 
#                          in taskuri separate (separate). Recomandat "single" pe FEP
#                          pentru a rula N instante pe acelasi GPU alocat concomitent.
#                          Default: separate.
#   --model MODEL          Model Ollama (default: preia din settings/.env)
#   --no-cache             Dezactivează cache LLM
#   --fresh                Șterge cache înainte de rulare
#   --blueprint-only       Doar faza 1+2 (skip populare KG)
#   --data-dir DIR         Director bază pentru rulări pe FEP
#                          (default: data/fep)
#   --input FILE           Specifică fișierul de intrare din data/input (ex: my_doc.pdf)
#   --resume DIR           Specifică un director anterior pentru reluare/cache (ex: data/fep/job_131117)
#   --chain N              Creează o secvență de N joburi dependente pentru un fișier mare
#   --time HH:MM:SS        Limita de timp SLURM (default: 06:00:00)
#   --partition PART       Partiție SLURM (default: gpu)
#   --project-dir DIR      Calea proiectului (default: ~/covaliu)
#   --sif PATH             Calea la ollama.sif
#                          (default: ~/ollama_project/ollama.sif)
#   --storage DIR          Calea la directorul de modele Ollama
#                          (default: ~/ollama_project/ollama_storage)
#   -h, --help             Afișează ajutor
#
# Exemple:
#   # 1 rulare
#   ./fep/launch.sh
#
#   # 5 rulări paralele
#   ./fep/launch.sh -n 5
#
#   # 3 rulări secvențiale cu cache șters
#   ./fep/launch.sh -n 3 --sequential --fresh
#
#   # 4 rulări paralele, blueprint-only, model mai mare
#   ./fep/launch.sh -n 4 --blueprint-only --model qwen2.5:14b
#
#   # Monitorizare după trimitere
#   watch -n 5 squeue -u $USER
# ============================================================

set -euo pipefail

# ── Valori implicite ──────────────────────────────────────────
N_RUNS=1
RUN_MODE="parallel"
TASKS_MODE="separate"
MODEL=""
EXTRA_ARGS=""
BASE_DATA_DIR="data/fep"
TIME_LIMIT="00:10:00"
PARTITION="dgxa100"
PROJECT_DIR="${HOME}/covaliu"
OLLAMA_SIF="${HOME}/ollama_project/ollama.sif"
OLLAMA_STORAGE="${HOME}/ollama_project/ollama_storage"

INPUT_FILE=""
RESUME_DIR=""
CHAIN_COUNT=0

# ── Parsare argumente ─────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input|-i)
            INPUT_FILE="$2"; shift 2 ;;
        --resume)
            RESUME_DIR="$2"; shift 2 ;;
        --chain)
            CHAIN_COUNT="$2"; TASKS_MODE="single"; shift 2 ;;
        -n|--runs)
            N_RUNS="$2"; shift 2 ;;
        --sequential)
            RUN_MODE="sequential"; shift ;;
        --parallel)
            RUN_MODE="parallel"; shift ;;
        --tasks)
            TASKS_MODE="$2"; shift 2 ;;
        --model)
            MODEL="$2"; shift 2 ;;
        --no-cache)
            EXTRA_ARGS="$EXTRA_ARGS --no-cache"; shift ;;
        --fresh)
            EXTRA_ARGS="$EXTRA_ARGS --fresh"; shift ;;
        --blueprint-only)
            EXTRA_ARGS="$EXTRA_ARGS --blueprint-only"; shift ;;
        --data-dir)
            BASE_DATA_DIR="$2"; shift 2 ;;
        --time)
            TIME_LIMIT="$2"; shift 2 ;;
        --partition)
            PARTITION="$2"; shift 2 ;;
        --project-dir)
            PROJECT_DIR="$2"; shift 2 ;;
        --sif)
            OLLAMA_SIF="$2"; shift 2 ;;
        --storage)
            OLLAMA_STORAGE="$2"; shift 2 ;;
        -h|--help)
            head -55 "$0" | grep '^#' | sed 's/^# \{0,2\}//'
            exit 0 ;;
        *)
            echo "Argument necunoscut: $1  (folosește --help)" >&2; exit 1 ;;
    esac
done

# ── Validare ──────────────────────────────────────────────────
if ! [[ "$N_RUNS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: -n trebuie să fie număr pozitiv, primit: $N_RUNS" >&2
    exit 1
fi

if [ ! -f "$OLLAMA_SIF" ]; then
    echo "ERROR: Ollama SIF nu există la: $OLLAMA_SIF" >&2
    echo "       Rulează mai întâi: cd ~/ollama_project && apptainer pull ollama.sif docker://ollama/ollama:latest" >&2
    exit 1
fi

# ── Sumar ─────────────────────────────────────────────────────
echo "============================================================"
echo "  AbA Ontology Pipeline — FEP Launcher (Apptainer)"
echo "============================================================"
echo "  Rulări       : $N_RUNS"
echo "  Execuție     : $RUN_MODE (secvential/paralel)"
echo "  Tip Taskuri  : TASKS_MODE = $TASKS_MODE (single=acelasi job, separate=array job)"
echo "  Model        : ${MODEL:-[Preia din env]}"
echo "  Argumente    : ${EXTRA_ARGS:-<niciunul>}"
echo "  Partiție     : $PARTITION"
echo "  Timp limită  : $TIME_LIMIT"
echo "  Director date: $BASE_DATA_DIR"
echo "  Proiect      : $PROJECT_DIR"
echo "  Ollama SIF   : $OLLAMA_SIF"
echo "  Modele dir   : $OLLAMA_STORAGE"
echo "  Input File   : ${INPUT_FILE:-<implicit>}"
echo "  Resume Dir   : ${RESUME_DIR:-<niciunul>}"
echo "  Chain Jobs   : ${CHAIN_COUNT}"
echo "============================================================"
echo ""

# ── Creare director de logs ───────────────────────────────────
mkdir -p "$PROJECT_DIR/fep/logs"

# ── Opțiuni comune SBATCH ────────────────────────────────────
EXPORT_VARS="ALL"
EXPORT_VARS="${EXPORT_VARS},PROJECT_DIR=${PROJECT_DIR}"
if [ -n "$MODEL" ]; then EXPORT_VARS="${EXPORT_VARS},OLLAMA_MODEL=${MODEL}"; fi
if [ -n "$INPUT_FILE" ]; then EXPORT_VARS="${EXPORT_VARS},INPUT_FILE=${INPUT_FILE}"; fi
if [ -n "$RESUME_DIR" ]; then EXPORT_VARS="${EXPORT_VARS},RESUME_DIR=${RESUME_DIR}"; fi
EXPORT_VARS="${EXPORT_VARS},OLLAMA_SIF=${OLLAMA_SIF}"
EXPORT_VARS="${EXPORT_VARS},OLLAMA_STORAGE=${OLLAMA_STORAGE}"
EXPORT_VARS="${EXPORT_VARS},EXTRA_ARGS=${EXTRA_ARGS}"
EXPORT_VARS="${EXPORT_VARS},TASKS_MODE=${TASKS_MODE},RUN_MODE=${RUN_MODE},N_RUNS=${N_RUNS},BASE_DATA_DIR=${BASE_DATA_DIR}"

SBATCH_OPTS=(
    "--job-name=onto-pipeline"
    "--partition=$PARTITION"
    "--nodes=1"
    "--ntasks=1"
    "--cpus-per-task=4"
    "--gres=gpu:1"
    "--mem=16G"
    "--time=$TIME_LIMIT"
    "--output=$PROJECT_DIR/fep/logs/%A_%a.out"
    "--error=$PROJECT_DIR/fep/logs/%A_%a.err"
)

# ── Trimitere joburi ──────────────────────────────────────────
if [ "$CHAIN_COUNT" -gt 0 ]; then
    echo "Trimit $CHAIN_COUNT joburi înlanțuite (CHAIN)..."
    BASE_CHAIN_DIR="${RESUME_DIR:-}"
    PREV_JOB=""
    
    for i in $(seq 1 "$CHAIN_COUNT"); do
        # Construim opțiunile pentru fiecare job din lanț pentru a evita suprascrierea greșită a vectorilor
        CUR_EXPORT="${EXPORT_VARS}"
        if [ -n "$BASE_CHAIN_DIR" ]; then
            CUR_EXPORT="${CUR_EXPORT},RESUME_DIR=${BASE_CHAIN_DIR}"
        fi
        
        RUN_OPTS=(
            "--job-name=onto-pipeline"
            "--partition=$PARTITION"
            "--nodes=1"
            "--ntasks=1"
            "--cpus-per-task=4"
            "--gres=gpu:1"
            "--mem=16G"
            "--time=$TIME_LIMIT"
            "--output=$PROJECT_DIR/fep/logs/%A_%a.out"
            "--error=$PROJECT_DIR/fep/logs/%A_%a.err"
            "--array=0"
            "--export=${CUR_EXPORT}"
        )

        if [ -n "$PREV_JOB" ]; then
            RUN_OPTS+=("--dependency=afterany:$PREV_JOB")
        fi

        # Daca acesta este primul job si nu avem un director setat manual, interceptam ID-ul primului job
        if [ -z "$BASE_CHAIN_DIR" ] && [ "$i" -eq 1 ]; then
            JOB_ID=$(sbatch "${RUN_OPTS[@]}" "$PROJECT_DIR/fep/job.sh" | awk '{print $NF}')
            BASE_CHAIN_DIR="$PROJECT_DIR/$BASE_DATA_DIR/job_${JOB_ID}"
            echo "  [Chain $i] → Job $JOB_ID (A creat bază date: $BASE_CHAIN_DIR)"
        else
            JOB_ID=$(sbatch "${RUN_OPTS[@]}" "$PROJECT_DIR/fep/job.sh" | awk '{print $NF}')
            echo "  [Chain $i] → Job $JOB_ID (Depinde de: $PREV_JOB, foloseste $BASE_CHAIN_DIR)"
        fi
        PREV_JOB="$JOB_ID"
    done
    echo ""
    echo "  Lansarea în lanț finalizată."

elif [ "$TASKS_MODE" = "single" ]; then
    SBATCH_OPTS+=(
        "--array=0"
        "--export=${EXPORT_VARS}"
    )
    JOB_ID=$(sbatch "${SBATCH_OPTS[@]}" "$PROJECT_DIR/fep/job.sh" | awk '{print $NF}')
    echo "Trimis mod SINGLE-TASK intern (Rulari: $N_RUNS | Stil: $RUN_MODE) → Job $JOB_ID"
    echo "  Director rezultate: $BASE_DATA_DIR/job_${JOB_ID}/run_{i}"
    echo "  Log principal: $PROJECT_DIR/fep/logs/${JOB_ID}_0.out"

elif [ "$N_RUNS" -eq 1 ]; then
    SBATCH_OPTS+=(
        "--array=0"
        "--export=${EXPORT_VARS}"
    )
    JOB_ID=$(sbatch "${SBATCH_OPTS[@]}" "$PROJECT_DIR/fep/job.sh" | awk '{print $NF}')
    echo "Trimis 1 rulare → Job $JOB_ID"
    echo "  Date : $BASE_DATA_DIR/job_${JOB_ID}_task_0"
    echo "  Log  : $PROJECT_DIR/fep/logs/${JOB_ID}_0.out"

elif [ "$RUN_MODE" = "parallel" ]; then
    # PARALEL: toate ca un singur job array
    LAST=$((N_RUNS - 1))
    SBATCH_OPTS+=("--array=0-${LAST}")
    SBATCH_OPTS+=("--export=${EXPORT_VARS}")
    JOB_ID=$(sbatch "${SBATCH_OPTS[@]}" "$PROJECT_DIR/fep/job.sh" | awk '{print $NF}')
    echo "Trimise $N_RUNS rulări paralele → Array Job $JOB_ID"
    echo ""
    for i in $(seq 0 "$LAST"); do
        echo "  Rulare $((i+1)) → date: $BASE_DATA_DIR/job_${JOB_ID}_task_${i}  |  log: fep/logs/${JOB_ID}_${i}.out"
    done

else
    # SECVENTIAL: fiecare job pornește după ce precedentul termină
    echo "Trimit $N_RUNS rulări secvențiale..."
    PREV_JOB=""
    for i in $(seq 0 $((N_RUNS - 1))); do
        RUN_OPTS=("${SBATCH_OPTS[@]}")
        RUN_OPTS+=("--array=$i")
        RUN_OPTS+=("--export=${EXPORT_VARS}")

        if [ -n "$PREV_JOB" ]; then
            RUN_OPTS+=("--dependency=afterok:$PREV_JOB")
        fi

        JOB_ID=$(sbatch "${RUN_OPTS[@]}" "$PROJECT_DIR/fep/job.sh" | awk '{print $NF}')
        echo "  Rulare $((i+1)) → Job $JOB_ID  |  date: $BASE_DATA_DIR/job_${JOB_ID}_task_${i}  |  log: fep/logs/${JOB_ID}_${i}.out"
        PREV_JOB="$JOB_ID"
    done
fi

# ── Hint monitorizare ─────────────────────────────────────────
echo ""
echo "Monitorizare:"
echo "  squeue -u \$USER                # Joburi în curs/așteptare"
echo "  watch -n 5 squeue -u \$USER     # Live view"
echo "  tail -f fep/logs/*.out          # Urmărire live log-uri"
echo ""
echo "Colectare rezultate (de pe laptop, după terminare):"
echo "  rsync -avz stefan.covaliu@fep.grid.pub.ro:~/covaliu/$BASE_DATA_DIR/ ~/covaliu/data/fep_results/"
echo ""
