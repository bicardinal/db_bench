
restart_qdrant:
	sudo docker compose -f docker-compose.qdrant.yml down
	sudo rm -rf qdrant_data
	mkdir -p qdrant_data
	sudo docker compose -f docker-compose.qdrant.yml up -d
