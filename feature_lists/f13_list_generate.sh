for i in $(seq -f %g 2.6 0.01 3); do
	sed 's/_2.pickle/_'$i'.pickle/g' f13_template.txt > 'f13_Rc'$i'.txt'
done
