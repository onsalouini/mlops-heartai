
.PHONY: install format lint security prepare train evaluate save load all ci build run push clean test-api

install:
	pip install -r requirements.txt

format:
	black model_pipeline.py main.py

lint:
	flake8 model_pipeline.py main.py --max-line-length=100

security:
	bandit -r model_pipeline.py main.py

prepare:
	python3 main.py --prepare

train:
	python3 main.py --train

evaluate:
	python3 main.py --evaluate

save:
	python3 main.py --save

load:
	python3 main.py --load

test:
	pytest test_pipeline.py -v

ci: format lint security test

all: install ci prepare train evaluate save

# ============================================
# Commandes Docker pour l'atelier 6
# ============================================

# Variables Docker
IMAGE_NAME = prenom_nom_classe_mlops
DOCKER_USER = $(shell whoami)
PORT = 8000
CONTAINER_NAME = heartai

# Construire l'image Docker
build:
	docker build -t $(IMAGE_NAME) .

# Lancer le conteneur
run:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true
	docker run -d -p $(PORT):$(PORT) --name $(CONTAINER_NAME) $(IMAGE_NAME)
	@echo "✅ API disponible sur http://localhost:$(PORT)"
	@echo "📚 Documentation: http://localhost:$(PORT)/docs"

# Tester l'API
test-api:
	@echo "🧪 Test health endpoint..."
	curl -s http://localhost:$(PORT)/health | python3 -m json.tool
	@echo "\n🧪 Test prediction endpoint..."
	curl -s -X POST http://localhost:$(PORT)/predict \
		-H "Content-Type: application/json" \
		-d '{"features": [63,1,3,145,233,1,0,150,0,2.3,0,0,1]}' | python3 -m json.tool

# Voir les logs
logs:
	docker logs -f $(CONTAINER_NAME)

# Arrêter le conteneur
stop:
	docker stop $(CONTAINER_NAME) || true

# Supprimer le conteneur
rm: stop
	docker rm $(CONTAINER_NAME) || true

# Nettoyer tout (conteneur + image)
clean: rm
	docker rmi $(IMAGE_NAME) || true

# Taguer l'image pour Docker Hub
tag:
	docker tag $(IMAGE_NAME) $(DOCKER_USER)/$(IMAGE_NAME):latest

# Se connecter à Docker Hub
login:
	docker login

# Pousser l'image sur Docker Hub
push: tag
	docker push $(DOCKER_USER)/$(IMAGE_NAME):latest
	@echo "✅ Image poussée sur Docker Hub: $(DOCKER_USER)/$(IMAGE_NAME):latest"

# Tout faire : construire, lancer, tester
all-docker: build run test-api

# Aide Docker
help-docker:
	@echo "📦 Commandes Docker disponibles:"
	@echo "  make build       - Construire l'image Docker"
	@echo "  make run         - Lancer le conteneur"
	@echo "  make test-api    - Tester l'API"
	@echo "  make logs        - Voir les logs du conteneur"
	@echo "  make stop        - Arrêter le conteneur"
	@echo "  make rm          - Supprimer le conteneur"
	@echo "  make clean       - Supprimer conteneur + image"
	@echo "  make login       - Se connecter à Docker Hub"
	@echo "  make tag         - Taguer l'image pour Docker Hub"
	@echo "  make push        - Pousser l'image sur Docker Hub"
	@echo "  make all-docker  - Construire + lancer + tester"

test:
pytest tests/ -v

test-coverage:
pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html
@echo "Rapport HTML : htmlcov/index.html"

test-api:
pytest tests/test_api.py -v

test-pipeline:
pytest tests/test_pipeline.py -v
