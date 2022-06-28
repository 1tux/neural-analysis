declare -a ego_sig_cells
declare -a allo_sig_cells
# ego_sig_cells=( 57 58 61 64 68 72 73 78 133 144 145 148 149 150 151 153 159 170 171 210 215 390 419 420 )
ego_sig_cells=( 16 20 22 50 52 57 58 61 64 68 72 73 78 82 85 87 93 94 95 97 98 99 133 144 145 148 149 150 151 153 159 170 171 200 201 202 204 205 210 215 390 419 420 )
allo_sig_cells=( 3 15 20 22 31 57 58 61 64 72 73 83 85 87 93 144 147 153 170 171 202 205 215 380 390 418 420 428 )
length=${#ego_sig_cells[@]}

for (( i=0; i < ${length}; i++ ))
do
  echo $i ${ego_sig_cells[$i]}
  bsub -R rusage[mem=20000] python main.py ${ego_sig_cells[$i]} shapley cache-path cache/${ego_sig_cells[$i]} no-plot xval no-cache
done

length2=${#allo_sig_cells[@]}
for (( i=0; i < ${length2}; i++ ))
do
  echo $i ${allo_sig_cells[$i]}
  bsub  -R rusage[mem=20000] python main.py ${allo_sig_cells[$i]} shapley cache-path cache/${allo_sig_cells[$i]} no-plot xval no-cache
done

#python main.py 72 shapley cache-path cache/72 no-plot xval
