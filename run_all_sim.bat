@echo off
setlocal enabledelayedexpansion
for /l %%y in (1, 1, 7) do (
	SET /a a=1000 * %%y
	SET /a z=!a! + 19

	for /l %%x in (!a!, 1, !z!) do (
		@echo %%x
		python main.py %%x no-plot shuffles 0 >> logs/log4.txt
		python dic.py %%x 10 >> logs/dic_log4.txt
	)
)
pause