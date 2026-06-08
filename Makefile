# Single entry point for the Call Center AI Agentic Pipeline.
#
# Common usage:
#   make build              # build cc-pipeline:cpu (default)
#   make build TORCH=gpu    # build cc-pipeline:gpu (cu130 torch)
#   make up                 # start postgres
#   make smoke              # 3-file smoke batch
#   make run ARGS="--workers 4 --batch-name nightly"
#   make psql               # open psql in the db container
#   make shell              # bash shell in the app image
#   make down               # stop everything (keeps DB)
#   make clean              # stop + wipe DB volume
#   make push REGISTRY=... TAG=...
#
# See `make help` for the full list.

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TORCH         ?= cpu
TORCH_VARIANT  = $(if $(filter gpu,$(TORCH)),cu130,cpu)
IMAGE         ?= cc-pipeline:$(TORCH)
REGISTRY      ?=
TAG           ?= $(TORCH)-$(shell git rev-parse --short HEAD 2>/dev/null || echo latest)
COMPOSE       ?= docker compose --env-file .env

# Export IMAGE so compose interpolates ${IMAGE} from this Makefile.
export IMAGE

.PHONY: help build smoke run up down clean shell logs psql push

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
build: ## Build image (TORCH=cpu|gpu, default cpu)
	docker build \
	    --build-arg TORCH_VARIANT=$(TORCH_VARIANT) \
	    -t $(IMAGE) .
	@echo
	@echo "Built $(IMAGE)"
	@docker images --format 'table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}' | grep -E 'REPOSITORY|cc-pipeline' || true

push: ## Tag and push the image (REGISTRY=... TAG=...)
	@test -n "$(REGISTRY)" || (echo "ERROR: REGISTRY is required" >&2 && exit 1)
	docker tag  $(IMAGE) $(REGISTRY)/cc-pipeline:$(TAG)
	docker push $(REGISTRY)/cc-pipeline:$(TAG)
	@echo "Pushed $(REGISTRY)/cc-pipeline:$(TAG)"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
up: ## Start postgres (detached)
	$(COMPOSE) up -d db
	$(COMPOSE) ps

smoke: ## Run a 3-file batch
	$(COMPOSE) run --rm app --limit 3 --batch-name smoke

run: ## Run a batch with custom args (ARGS="--workers 4 --batch-name nightly")
	$(COMPOSE) run --rm app $(ARGS)

shell: ## Open a bash shell in the app image
	$(COMPOSE) run --rm --entrypoint bash app

# ---------------------------------------------------------------------------
# Inspect / teardown
# ---------------------------------------------------------------------------
logs: ## Tail compose logs
	$(COMPOSE) logs -f --tail=200

psql: ## Open psql in the db container
	$(COMPOSE) exec db sh -c 'psql -U "$${POSTGRES_USER:-cc_pipeline}" -d "$${POSTGRES_DB:-call_center}"'

down: ## Stop services (keeps DB volume)
	$(COMPOSE) down

clean: ## Stop services AND wipe the DB volume
	$(COMPOSE) down -v
