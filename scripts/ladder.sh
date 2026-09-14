#!/bin/bash
# ladder.sh <run_name> [nar_lora.pt]
# Renders LADDER_LYRICS with the style caption of LADDER_STYLE_TRACK (a track name under /workspace/real/artist) from every <run>/step-N.pt,
# then, if FINALS is set, one render per line of that file ("tag style_track lyrics_file seed") from <run>/best.pt.
export HF_HOME=/workspace/hf; PY="/workspace/yue2venv/bin/python -u"; cd /workspace/tok/full; RUN=$1; NAR=${2:-nar_lora_joint_v4.pt}
: "${LADDER_STYLE_TRACK:?set LADDER_STYLE_TRACK}"; : "${LADDER_LYRICS:?set LADDER_LYRICS (path to a tagged lyrics .txt)}"; SEED=${LADDER_SEED:-12}
for ck in $(ls $RUN/step-*.pt | sort -t- -k2 -n); do N=$(basename $ck .pt | cut -d- -f2)
  $PY ar_generate.py $ck $NAR ${RUN}_step${N}_ladder $LADDER_STYLE_TRACK $LADDER_LYRICS $SEED 2>&1 | grep --line-buffered -E "GEN DONE|Traceback|Error"
done
if [ -n "$FINALS" ]; then while read -r tag st ly sd; do [ -z "$tag" ] && continue
  $PY ar_generate.py $RUN/best.pt $NAR ${tag}_${RUN#ar_lora_} $st $ly $sd 2>&1 | grep --line-buffered -E "GEN DONE|Traceback|Error"; done < "$FINALS"; fi
