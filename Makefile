.PHONY: help build up down logs reset-db clean test dev populate-db

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build all Docker images
	docker-compose build

up: ## Start all services
	docker-compose up -d

down: ## Stop all services
	docker-compose down

logs: ## View logs from all services
	docker-compose logs -f

reset-db: ## Reset database with sample data
	docker exec -it stackai-api python scripts/reset_db.py

populate-db: ## Populate database with sample data (my master's thesis :))
	HOST=localhost ./scripts/populate_db.sh

clean: ## Remove all containers, images, and volumes
	docker-compose down -v --rmi all --remove-orphans

test: ## Run tests
	docker-compose -f docker-compose.test.yaml up --build --exit-code-from app-test

dev: ## Start development environment
	docker-compose up -d postgres
	@echo "Postgres started. Run 'make reset-db' to populate with sample data."
	@echo "Then start your local API server." 