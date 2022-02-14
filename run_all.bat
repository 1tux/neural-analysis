@echo off
for /l %%x in (1, 1, 200) do (
	@echo %%x
	python main.py %%x no-plot >> logs/log2.txt
	python dic.py %%x 10 >> logs/dic_log2.txt
)
pause