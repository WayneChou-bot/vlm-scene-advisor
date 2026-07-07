#!/usr/bin/env bash
# Assemble Blender frames (rendered every 3rd frame of a 24fps timeline = 8fps)
# into a denoised, upscaled, motion-interpolated 24fps clip.
# usage: assemble_3d_clip.sh <frames_dir> <out.mp4>
set -e
DIR=${1:-render/anim}; OUT=${2:-data/warehouse_3d.mp4}
ffmpeg -y -loglevel error -framerate 8 -pattern_type glob -i "$DIR/frame_*.png" \
  -vf "hqdn3d=3:2:4:3,scale=960:540:flags=lanczos,minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc:vsbmc=1" \
  -c:v libx264 -pix_fmt yuv420p -crf 19 "$OUT"
echo "wrote $OUT"
