install:
	mkdir -p /goinfre/$(USER)/venvs/call_me_maybe
	mkdir -p /goinfre/$(USER)/.uv_cache
	UV_PROJECT_ENVIRONMENT=/goinfre/$(USER)/venvs/call_me_maybe \
	UV_CACHE_DIR=/goinfre/$(USER)/.uv_cache \
	uv sync
run:
	HF_HOME=/goinfre/$(USER)/huggingface \
	TRANSFORMERS_CACHE=/goinfre/$(USER)/huggingface \
	HUGGINGFACE_HUB_CACHE=/goinfre/$(USER)/huggingface \
	UV_PROJECT_ENVIRONMENT=/goinfre/$(USER)/venvs/call_me_maybe \
	UV_CACHE_DIR=/goinfre/$(USER)/.uv_cache \
	uv run python3 -m src
debug:
	HF_HOME=/goinfre/$(USER)/huggingface \
	UV_PROJECT_ENVIRONMENT=/goinfre/$(USER)/venvs/call_me_maybe \
	uv run python3 -m pdb src/__main__.py
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
lint:
	HF_HOME=/goinfre/$(USER)/huggingface \
	UV_PROJECT_ENVIRONMENT=/goinfre/$(USER)/venvs/call_me_maybe \
	flake8 .
	uv run mypy . --warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs
lint-strict:
	HF_HOME=/goinfre/$(USER)/huggingface \
	UV_PROJECT_ENVIRONMENT=/goinfre/$(USER)/venvs/call_me_maybe \
	flake8 .
	uv run mypy . --strict