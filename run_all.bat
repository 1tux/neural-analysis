@echo off
for /l %%x in (1, 1, 230) do (
	@echo %%x
	python main.py %%x no-plot >> log.txt
)
pause