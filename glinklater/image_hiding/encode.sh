#! /bin/bash
# echo $* | python hide.py encode ~/Pictures/Tesseract1920.png - encoded.png
cat - | python hide.py encode $1 - encoded.png