ls inputs/shuffles/ | awk '{split($0,a,"_"); print "bsub","-R \"rusage[mem=20000]\"","\"python","main.py",a[1],"shuffle",a[2],"no-plot","no-cache","xval","cache-path","cache/"a[1],"\""}' > run_shuffles.sh