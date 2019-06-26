#!/usr/bin/env bash

input_path=$1
output_path=${input_path%%.*}

mkdir output_path
ffmpeg -i $input_path -vf "select=not(mod(n\,5))" -vsync vfr -qscale:v 2 -start_number 0 output_path/%05d.jpg
