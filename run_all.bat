@echo off
for /l %%x in (1, 1, 430) do (
	@echo %%x
	python main.py %%x no-plot >> logs/log5.txt
	python dic.py %%x 10 >> logs/dic_log5.txt
)
pause