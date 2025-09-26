### Purpose and Scope

ChatFlow Miner targets data scientists analysing omnichannel chat logs. Users upload CSV event logs, explore cases through interactive filters, derive per-case aggregations, and materialize Directly-Follows Graph (DFG) models to study chatbot/customer interactions. The scope is limited to client-side Streamlit execution with pm4py-backed discovery; there is no persistent backend or multi-project workspace.

### Architecture and Major Components

The Streamlit page in `chatflow_miner/app/dashboard.py` wires sidebar controls, file upload, model selection, and renders saved models. Core logic lives under `chatflow_miner/lib`: `state.manager` owns Streamlit `session_state`; `inputs.dataset` handles uploads; `utils.load`/`verify` normalise CSV data; `event_log.view` offers lazy views with composable `BaseFilter` implementations from `filters`; `aggregations` provides auxiliary preprocessing, per-case aggregators, and registry utilities; `process_models` wraps pm4py DFG computation plus visualization, with `ProcessModelRegistry` managing named models.

### Entry Points and Interaction Flow

Run the dashboard with `streamlit run chatflow_miner/app/dashboard.py`. The UI initialises session state, offers a dialog (`input_dataset`) to upload CSV logs, applies optional filters (`filter_section`, composed of fragments like `filter_by_agents`/`filter_by_variants`), lets users trigger model generation (`generate_process_model` → `ProcessModelView`), and displays selected saved models via `render_saved_model_ui`. There are no REST endpoints, CLIs, or background schedulers.

### Public Interfaces and Contracts

`initialize_session_state`, `open_input_dialog`, `set_log_eventos`, and `get_log_eventos`, and related helpers maintain UI state, with the sentinel placeholder `"Criar novo modelo de processo..."`. `EventLogView(base_df, filters)` exposes `.filter(...)`, `.compute() -> pd.DataFrame`, `.head()`, and `.to_csv(path)`; filters must return boolean `pd.Series` aligned with the original index. `ProcessModelRegistry` behaves like a mutable mapping of `str -> ProcessModelView|None`, enforcing placeholder rules and offering helpers like `.add_many`, `.compute_map`, and `.to_graphviz_map`. `ProcessModelView(log_view, model)` lazily materialises models via `.compute()` and caches Graphviz renderings (`.to_graphviz(**kwargs)`). Aggregation contracts rely on `CaseAggView.with_aux(...).with_aggregator(...)` returning `Dict[case_id, Any]`, with `CaseVariantAggregator` producing `VariantInfo` records and `CaseDateAggregator` yielding ISO dates.

### Data Models and Storage Schema

Event data is held in-memory as pandas DataFrames shaped by pm4py expectations, with required columns (`CASE_ID`, `ACTIVITY`, `START_TIMESTAMP`, `END_TIMESTAMP`, etc.) enforced through `verify_format`. `load_dataset` drops a `duration_seconds` column if present, parses timestamps, and calls `pm4py.format_dataframe`. DFG computation returns `(dfg, start_activities, end_activities)` tuples; Graphviz visualisations are pm4py `gviz` objects. There is no database or ORM, hence no migrations.

### Configuration and Environment Variables

The app relies on Streamlit configuration and session state only; there are no custom environment variables or config files. CSV delimiter selection comes from a UI `selectbox`. Caching is handled via object-level caches (`ProcessModelView`, Streamlit session) rather than environment toggles.

### Dependencies, Versions, and External Services

Key dependencies (see `requirements.txt`) include `streamlit==1.49.1`, `pm4py==2.7.11.14`, `pandas`, `graphviz`, and `pytest`. pm4py pulls in process mining algorithms; Graphviz must be installed system-wide for diagram rendering. No external APIs or databases are contacted beyond local file I/O.

### Build, Run, and Test Workflows

Install dependencies with `pip install -r requirements.txt`, then launch `streamlit run chatflow_miner/app/dashboard.py`. Run automated checks via `pytest` (tests live under `tests/lib/...`) or execute `python verify_tests.py` to ensure reorganised tests import correctly. Streamlit’s live-reload aids manual QA during development.

### Side Effects, I/O, and Error Handling

CSV uploads are read via `pandas.read_csv`; `EventLogView.to_csv` writes filtered data to disk when requested. pm4py discovery and visualization operate in-memory but may invoke Graphviz binaries. Session helpers mutate `st.session_state`. Recoverable issues surface through `st.error`, while functions like `load_dataset` call `st.stop()` after reporting errors. Filters and aggregators raise `MissingColumnsError`/`FilterError` for schema mismatches; registry operations raise `KeyError`/`ValueError` for invalid names. `CaseAggView` logs debug information through the standard `logging` module.

### Security and Privacy Considerations

The app has no authentication or authorisation layers; deployment should rely on Streamlit sharing/security mechanisms. Uploaded logs remain in process memory; avoid logging sensitive contents. Secrets management is out of scope—never hardcode credentials, and ensure external services (if added) pull secrets from secure sources.

### Performance, Scalability, and Concurrency

Streamlit runs synchronously per user session. `ProcessModelView` caches computed models and Graphviz outputs keyed by kwargs to prevent redundant pm4py work. Filters and aggregations copy DataFrames instead of mutating them, supporting idempotent recomputation. Large logs or dense DFGs may stress pm4py/Graphviz; `DFGModel.to_graphviz` exposes `max_num_edges` to cap rendering complexity. Session state isolates concurrent users; avoid global mutable state outside `st.session_state`.

### Edge Cases, Constraints, and Known Issues

`AgentFilter` only accepts `"ai"` or `"human"` (optionally keeping `"syst"` records); `TimeWindowFilter` requires `START_TIMESTAMP` and handles missing end timestamps by mirroring start times. Model registry placeholders are only valid on an empty registry. `render_saved_model_ui` catches and reports visualization failures. The filters UI is marked “Em construção”, signalling incomplete functionality. Graphviz or pm4py import failures will prevent model discovery/rendering; tests mock these dependencies to keep unit tests light.

### Extension Points and Contribution Guidelines

Add new filters by subclassing `BaseFilter` and composing them via `EventLogView.filter`; expose UI controls as `@st.fragment` components similar to `filter_by_agents`. Implement new aggregations by subclassing `BaseCaseAggregator`, optionally registering them with `register_aggregator`. Additional process models should subclass `BaseProcessModel` and integrate with `ProcessModelView`/`ProcessModelRegistry`. When expanding state, extend `initialize_session_state` and provide accessor helpers to keep Streamlit session usage consistent. Use existing logging and error-reporting conventions, respect lazy evaluation patterns, and write corresponding tests under `tests/lib/...` to validate new behaviours.