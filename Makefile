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

# CPU vs GPU is a Compose profile. `make serve TORCH=gpu` flips the server,
# its image, and the GPU device reservation together. Default is CPU.
PROFILE        = $(if $(filter gpu,$(TORCH)),gpu,cpu)
BATCH_PROFILE  = $(if $(filter gpu,$(TORCH)),gpu-batch,cpu-batch)
SERVER_SVC     = $(if $(filter gpu,$(TORCH)),server-gpu,server)
APP_SVC        = $(if $(filter gpu,$(TORCH)),app-gpu,app)

.PHONY: help build smoke run up down clean shell logs psql push serve serve-down serve-logs serve-shell server-status vllm-gpu vllm-cpu vllm-down vllm-logs

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

smoke: ## Run a 3-file batch (TORCH=gpu for GPU)
	$(COMPOSE) --profile $(BATCH_PROFILE) run --rm $(APP_SVC) --limit 3 --batch-name smoke

run: ## Run a batch with custom args (ARGS="--workers 4 ..."; TORCH=gpu for GPU)
	$(COMPOSE) --profile $(BATCH_PROFILE) run --rm $(APP_SVC) $(ARGS)

shell: ## Open a bash shell in the app image
	$(COMPOSE) --profile $(BATCH_PROFILE) run --rm --entrypoint bash $(APP_SVC)

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

# ---------------------------------------------------------------------------
# Long-running HTTP server (alternative to one-shot batch runs)
# ---------------------------------------------------------------------------
serve: ## Start the HTTP server (detached, CPU). `make serve TORCH=gpu` for GPU
	$(COMPOSE) --profile $(PROFILE) up -d db $(SERVER_SVC)
	$(COMPOSE) --profile $(PROFILE) ps $(SERVER_SVC)
	@echo
	@echo "Health:  curl http://127.0.0.1:$${SERVER_HOST_PORT:-8000}/health"
	@echo "Submit:  curl -X POST -H 'Content-Type: application/json' \\"
	@echo "             -d '{\"audio_path\":\"audio_files/your.wav\"}' \\"
	@echo "             http://127.0.0.1:$${SERVER_HOST_PORT:-8000}/jobs"

serve-down: ## Stop the server (DB stays up — use `make down` to stop both)
	$(COMPOSE) --profile $(PROFILE) stop $(SERVER_SVC)
	$(COMPOSE) --profile $(PROFILE) rm -f $(SERVER_SVC)

serve-logs: ## Tail the server logs (TORCH=gpu for the GPU server)
	$(COMPOSE) --profile $(PROFILE) logs -f --tail=200 $(SERVER_SVC)

serve-shell: ## Open a shell in the running server container
	$(COMPOSE) --profile $(PROFILE) exec $(SERVER_SVC) bash

server-status: ## Quick status: health + recent jobs
	@curl -fsS http://127.0.0.1:$${SERVER_HOST_PORT:-8000}/health \
	    && echo \
	    && curl -fsS "http://127.0.0.1:$${SERVER_HOST_PORT:-8000}/jobs?limit=5" \
	    || echo "Server not reachable on :$${SERVER_HOST_PORT:-8000}"

# ---------------------------------------------------------------------------
# Qwen / vLLM backing service (separate file: compose.vllm.yaml)
# ---------------------------------------------------------------------------
vllm-gpu: ## Start Qwen vLLM on GPU (dev)
	$(COMPOSE) -f compose.vllm.yaml --profile gpu up -d
	@echo "vLLM (GPU) on http://127.0.0.1:$${VLLM_HOST_PORT:-8080}/v1"

vllm-cpu: ## Start Qwen vLLM on CPU (prod; build vllm-cpu:local first)
	$(COMPOSE) -f compose.vllm.yaml --profile cpu up -d
	@echo "vLLM (CPU) on http://127.0.0.1:$${VLLM_HOST_PORT:-8080}/v1"

vllm-down: ## Stop the vLLM service (CPU or GPU)
	$(COMPOSE) -f compose.vllm.yaml --profile gpu --profile cpu down

vllm-logs: ## Tail vLLM logs
	$(COMPOSE) -f compose.vllm.yaml logs -f --tail=200
