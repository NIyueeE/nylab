# Makefile
debug-up:
	docker-compose -f docker-compose.debug.yml up --build -d

debug-logs:
	docker-compose -f docker-compose.debug.yml logs -f

debug-down:
	docker-compose -f docker-compose.debug.yml down -v

attach-backend:
	docker attach nylab_backend_debug

build-up:
	docker-compose -f docker-compose.yml up --build -d

build-down:
	docker-compose -f docker-compose.yml down -v