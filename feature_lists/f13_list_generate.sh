for i in $(seq -f %g 0.5 0.1 4) 5 6 7; do
	sed 's/_2.pickle/_'$i'.pickle/g' f13_template.txt > 'f13_Rc'$i'.txt'
done
