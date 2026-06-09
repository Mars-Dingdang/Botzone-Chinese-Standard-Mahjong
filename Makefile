PYTHON ?= python3

.PHONY: install-train install-deploy test generate-bc-data train-bc train-ppo evaluate export-bot verify-deploy

install-train:
	$(PYTHON) -m pip install -r requirements/train.txt

install-deploy:
	$(PYTHON) -m pip install -r requirements/deploy.txt

test:
	$(PYTHON) -m unittest discover -s tests -v

generate-bc-data:
	$(PYTHON) scripts/generate_bc_data.py --games 100

train-bc:
	$(PYTHON) scripts/train_bc.py

train-ppo:
	$(PYTHON) scripts/train_ppo.py

evaluate:
	$(PYTHON) scripts/evaluate.py --games 400 --duplicate

export-bot:
	$(PYTHON) scripts/export_bot.py

verify-deploy:
	$(PYTHON) scripts/verify_deploy.py
