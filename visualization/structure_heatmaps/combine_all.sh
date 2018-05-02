# Execute on my machine
pdfunite $(ls structure_heatmap_*.pdf | sort -V | xargs echo) combined.pdf
