.PHONY: setup run-cli run-server clean

# Garantiza que la carpeta exista antes de cualquier comando de Docker
setup:
	mkdir -p data
	docker compose -f deployments/docker-compose.yaml --profile setup run --rm cli-generator

run-server:
	docker compose -f deployments/docker-compose.yaml up -d server

clean:
	rm -rf data/*.bin