#!/bin/bash
for i in {1..7}
do
	for j in {1000..1018}
	do
		x=$((i * 1000 + j))
	        bsub -e "logs/text2.txt" -o "logs/text.txt" "python main.py $x no-plot shapley cache-path cache/$x/ &>> logs/log$x.txt"
	done
done
