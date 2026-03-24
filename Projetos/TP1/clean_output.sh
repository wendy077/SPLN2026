#!/bin/bash


rm -f data/raw/*.txt
rm -f data/clean/*.txt
rm -f data/clean/*.json
rm -f output/tex/*.tex
rm -f output/pdf/*.pdf
rm -f output/pdf/*.aux
rm -f output/pdf/*.log
rm -f output/pdf/*.out

echo "Limpo. Prosseguir com run_pipeline.sh"