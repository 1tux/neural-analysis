@echo off
for /l %%x in (1, 1, 10) do (
	@echo %%x
	python main.py %%x no-plot
)
pause