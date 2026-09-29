#!/usr/bin/env bash
# Rebuild every asset of the walking loop from the source clip, then feed Remotion and HyperFrames.
#   usage: tools/loop_pipeline/run_pipeline.sh            (work files go to .loop_work/, ~30-40 min on 4 cores)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
TOOLS=$REPO/tools/loop_pipeline
WORK=${LOOP_WORK:-$REPO/.loop_work}
export LOOP_WORK=$WORK LOOP_REPO=$REPO PYTHONPATH=$TOOLS PYTHONWARNINGS=ignore
mkdir -p "$WORK"/{frames,masks,depth,clean,sprites,thumbs}
cd "$WORK"

# run a script over a frame range split into 4 parallel chunks: par <script> <first> <last> [extra args before range]
par() {
  local script=$1 a=$2 b=$3; shift 3
  local n=$(( (b - a + 4) / 4 )) s e
  for ((s=a; s<=b; s+=n)); do
    e=$(( s + n - 1 )); (( e > b )) && e=$b
    python3 "$TOOLS/$script" "$@" "$s" "$e" > "$WORK/log_${script%.py}_$s.txt" 2>&1 &
  done
  wait
}

echo "1/9 extract frames";            ffmpeg -loglevel error -y -i "$REPO/assets/source/gemini_generated_video.mp4" frames/f%04d.png
echo "2/9 camera path";               python3 "$TOOLS/campath.py" > log_campath.txt
echo "3/9 back-layer panorama";       python3 "$TOOLS/build_pano.py"
echo "4/9 wizard residual maps";      par wizmask.py 136 220
echo "5/9 wizard masks";              python3 "$TOOLS/selectmask.py"
echo "6/9 depth maps";                par depth.py 136 220
echo "7/9 clean plates";              python3 "$TOOLS/run_clean.py" hud 142 153; par run_clean.py 154 220 full; par postfill.py 154 220
echo "8/9 sprites";                   par sprites.py 154 220
echo "9/9 Remotion + HyperFrames";    python3 "$TOOLS/build_assets.py"; python3 "$TOOLS/build_hyperframes.py"
echo "done: render with  (cd remotion && npx remotion render WalkingLoop ../video/walking_loop.mp4)"
