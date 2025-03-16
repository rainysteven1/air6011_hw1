run:
	docker build -t  air6011_hw1_q1:latest . && docker compose up -d

debug-cifar:
	python main.py -d cifar -l torch

debug-cupy:
	python main.py -d mnist -l cupy

debug-mnist:
	python main.py -d mnist -l torch