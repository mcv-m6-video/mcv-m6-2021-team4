#!/usr/bin/env bash
ffmpeg -i vdo.avi -r 1/1 img/$filename%03d.bmp
