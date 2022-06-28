#!/bin/bash
for i in {1..430}
do
	bsub -e "logs/text2.txt" -o "logs/text.txt" "python main.py $i no-plot shapley cache-path cache/$i/ &>> logs/log$i.txt"
done
