# ChatFlow Miner – Agent Guide
This repository is a **Streamlit**‑based process mining tool. It lets data scientists upload event logs from chat interactions, explore them through composable filters, compute Directly‑Follows Graph (DFG) and Petri‑net models via **pm4py**, and visualise results directly in the browser. For a high‑level overview see the .
## Setup & environment
- **Python ≥3.8 is required**; create a virtual environment to avoid polluting system packages.
- The only configuration is the Python environment; there are **no custom env vars** or database secrets. Graphviz must be installed on the host to render diagrams.
```
# create and activate virtualenv
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# optional: verify graphviz is available
which dot  # should output a path
```

If pm4py or graphviz fail to install in your environment, see the open questions section below.
## Running the app
Run the Streamlit dashboard locally:
```
streamlit run chatflow_miner/app/dashboard.py --global.disableWidgetStateDuplicationWarning=1
```
- By default Streamlit serves on localhost:8501. You can set --server.port to change the port.
- Use the sidebar button “Carregar log de eventos” to upload a CSV of your chat events. Required columns include CASE_ID, ACTIVITY, START_TIMESTAMP and END_TIMESTAMP. Delimiters are selectable via the UI; timestamps must be parsable by pandas.
- You can find an example of input inside the root directory of this repository through the file `event_log_example.csv`. Use it whenever you run the app.
- The **“Modelos de processo”** select box lists all saved models. Selecting the placeholder **“Criar novo modelo de processo…”** exposes the filter panel, allowing you to compose filters before generating a new model.
## Tests & quality gates
Automated tests live under tests/ and use **pytest**. New code **must not break existing tests**. Run the suite from the repo root:
```
pytest -q
```

Tests cover filters (tests/lib/filters), aggregations (tests/lib/aggregations), event-log views (tests/lib/event_log), and process models (tests/lib/process_models). When adding new features, add corresponding tests alongside your code. Aim for at least one unit test per public function and avoid slow, external dependencies in unit tests. Use pytest.mark to distinguish integration tests if needed.

### Quality gates & CI workflow
Every push and pull request runs the **Quality Gates** workflow (`.github/workflows/quality-gates.yml`). Reproduce it locally before opening a PR:

1. Install dependencies:
   ```
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
2. Verify that the reorganised tests are importable:
   ```
   python verify_tests.py
   ```
3. Execute the targeted coverage suite (produces `coverage.xml`):
   ```
   pytest -q tests/lib/filters tests/lib/aggregations tests/lib/event_log tests/lib/process_models \
     --strict-config --strict-markers -W error \
     --cov=chatflow_miner.lib --cov-report=xml:coverage.xml --cov-report=term
   ```
4. Run the fail-fast test pass for the remaining suite:
   ```
   pytest -q --maxfail=1 --strict-config --strict-markers -W error
   ```
5. Generate the Ruff report (writes `ruff_report.json`):
   ```
   ruff check chatflow_miner tests --output-format json --exit-zero > ruff_report.json
   ```
6. Compare the freshly generated reports against the stored baselines:
   ```
   python scripts/check_quality_gates.py \
     --baseline quality_gate_baselines.yml \
     --coverage coverage.xml \
     --ruff ruff_report.json
   ```

The baseline thresholds live in `quality_gate_baselines.yml`:

- `coverage.chatflow_miner.lib`: minimum overall coverage for the `chatflow_miner.lib` package (currently **62.45%**).
- `ruff.violation_count`: maximum number of Ruff lint findings allowed across `chatflow_miner` and `tests` (currently **28**).

With every proposed change you make, recalculate the quality-gate metrics for the codebase.
If any of these metrics change, update the values in the YML file. Avoid worsening these metrics at all costs.

The workflow uploads both `coverage.xml` and `ruff_report.json` as artifacts so reviewers can inspect the raw metrics.

#### Refreshing baselines after improvements
If you legitimately raise coverage or reduce Ruff violations:

1. Rerun steps 3 and 5 above to create fresh reports that reflect the improved state.
2. Inspect the new totals:
   - Coverage: run `python - <<'PY'` with an `xml.etree.ElementTree` parser (see `scripts/check_quality_gates.py`) or reuse that script’s summary output.
   - Ruff violations: `python -c "import json; print(len(json.load(open('ruff_report.json'))))"`.
3. Update `quality_gate_baselines.yml` with the improved values (never decrease them).
4. Commit the updated baseline alongside the code changes so CI enforces the new thresholds going forward.

Temporary artifacts such as `coverage.xml` and `ruff_report.json` are ignored via `.gitignore`, but double-check before committing.

## Code layout & architecture
- **chatflow_miner/app/** contains Streamlit entry points (dashboard.py). Only UI logic should live here.
- **chatflow_miner/lib/** holds pure Python modules for data loading (inputs/dataset.py), state management (state/manager.py), filtering (filters/…), aggregations (aggregations/…), and process models (process_models/…). Keep these modules free of Streamlit calls; they should be deterministic and easy to unit‑test.
- **Filters** must subclass BaseFilter and implement a __call__(df) -> pd.Series[bool]. Register new filters in filters/streamlit_fragments.py to expose UI widgets.
- **Aggregations** subclass BaseCaseAggregator and should not mutate input data. Add them to aggregations/registry.py if they should appear in the UI.
- **Process models** subclass BaseProcessModel (e.g. DFGModel, PetriNetModel). Expose Graphviz renderings via to_graphviz(**kwargs) and register them in the model registry.
- **State management** lives in chatflow_miner/lib/state/manager.py. Use the provided initialize_session_state and helper functions instead of directly manipulating st.session_state.
### Do
- Write pure functions in lib/; UI code in app/ should orchestrate those functions.
- Use environment variables (e.g. GRAPHVIZ_BIN) to override defaults rather than hardcoding paths.
- Validate input data early using verify_format and raise FilterError/MissingColumnsError when schemas are incorrect.
- Cache expensive computations (e.g. pm4py model discovery) by leveraging the caches provided in ProcessModelView and st.session_state.
### Don’t
- Don’t store secrets or PII in the codebase. Uploaded logs remain only in memory; they are not persisted.
- Don’t mutate pandas DataFrames in place; always return a copy when filtering or aggregating.
- Don’t import Streamlit into library modules; this couples business logic to the UI and makes testing harder.
## Common tasks
**Add a new filter** 1. Create a subclass of BaseFilter in chatflow_miner/lib/filters/. 2. Implement __call__(self, df: pd.DataFrame) -> pd.Series and a factory fragment() in filters/streamlit_fragments.py to add the UI widget. 3. Write unit tests under tests/lib/filters/.
**Add a new aggregator** 1. Subclass BaseCaseAggregator and implement __call__(self, df: pd.DataFrame) -> Dict[CaseId, Any]. 2. Optionally implement with_aux if your aggregator needs auxiliary computations. 3. Register it in aggregations/registry.py and add a test.
**Add a new process model** 1. Subclass BaseProcessModel and implement compute(self, df) -> Any and to_graphviz. 2. Expose a UI button in chatflow_miner/lib/process_models/ui.py and update the factory map in state/manager.py.
## Security & data handling
The app has no authentication layer. Deploy behind Streamlit’s built‑in access controls (e.g. password protection). Treat uploaded chat logs as sensitive: never log or persist them to disk without explicit user consent. Configuration and secrets (if you add external services) must be injected via environment variables or secret managers; **never hard‑code API keys**. Follow least‑privilege when accessing files.
## CI/CD & PR process
There is currently no CI pipeline. Agents should emulate a CI run locally: ensure pytest passes and optional linters/type checks succeed before opening a pull request.
Branch names should be descriptive (feat/filter-agent-type, fix/petri-net-bug). Commit messages should follow the  style (feat:/fix:/refactor:) to aid changelog generation. Include a brief summary and, if relevant, a reference to an issue.
When submitting a PR, include:
- A summary of the change and reasoning.
- Evidence that tests and quality gates pass (copy of local run output).
- Updates to this AGENTS.md or the README if behaviour or commands change.
## Gotchas & pitfalls
- **Graphviz dependency**: If diagrams do not render, ensure the dot executable is installed and available in the PATH or set GRAPHVIZ_BIN.
- **Large logs**: pm4py’s DFG and Petri‑net discovery can be expensive on large datasets. Use filtering and aggregation to reduce size, and limit rendering complexity via max_num_edges in to_graphviz.
- **Streamlit caching**: Changing module code might not invalidate Streamlit’s cache. Restart the Streamlit server (Ctrl+C and re‑run) after modifying library code.
- **Incomplete UI**: The filter section labelled “Em construção” in the sidebar indicates work‑in‑progress. Avoid exposing unfinished fragments without tests.
## Glossary & references
- **Case/Case ID** – A conversation session; used as the grouping key for aggregations.
- **DFG (Directly‑Follows Graph)** – A model capturing transitions between activities; implemented in DFGModel.
- **Petri net** – A process model representing concurrency; implemented in PetriNetModel.
- **ProcessModelRegistry** – A mapping of model names to ProcessModelView objects; persists only in Streamlit session.
- **pm4py** – Library used for process mining algorithms.
For more guidance on SaaS engineering best practices (configuration via environment, test‑driven development, and continuous deployment) refer to the *Engineering Software as a Service* text included in this repo.
