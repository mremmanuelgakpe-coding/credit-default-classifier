.PHONY: install train train-all test api demo docker-build docker-run

install:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

train:
	PYTHONPATH=src .venv/bin/python -m credit_default.train --feature-set full

train-all: train
	PYTHONPATH=src .venv/bin/python -m credit_default.train --feature-set application

test:
	.venv/bin/python -m pytest -q

api:
	set -a; . ./.env; set +a; PYTHONPATH=src .venv/bin/uvicorn api.main:app --reload --port 8000

docker-build:
	docker build -t credit-default-api:1.0.0 .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env --read-only --cap-drop ALL \
	  --security-opt no-new-privileges --tmpfs /tmp credit-default-api:1.0.0

demo:
	.venv/bin/streamlit run demo/app.py --server.address 127.0.0.1
