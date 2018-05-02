# unite all pdfs from networks
# usage (run on my box): . unite_pdfs.sh '1x5_z-score/nn_map_*.pdf' '1x5_z-score/united.pdf'
list=$(ls $1 | sort -V)
list=$list" ""$2"
echo $list
pdfunite $list
