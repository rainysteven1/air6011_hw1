build:
	docker build -t  air6011_hw1_q1:latest .

run:
	docker compose up -d

debug-cifar:
	python main.py --dataset cifar --device cuda:0

debug-mnist:
	python main.py --dataset mnist --device cuda:0