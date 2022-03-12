@echo off
for /l %%x in (3, 1, 10) do (
	@echo %%x
	python main.py %%x shapley no-plot cache-path cache/%%x/ >> logs/log6.txt
	python dic.py %%x 10 >> logs/dic_log6.txt
)
pause