# SChain Graph Explorer

A local web app for exploring supply-chain entities, relationships, and their supporting sources.

## Run

- **Windows:** Double-click `start_server.bat`.
- **macOS / Linux:** Run `./start_server.sh`.
- **Alternative:** Run `python server.py --port 8080 --no-browser` and open `http://127.0.0.1:8080`.

Python 3.10 or newer is required. The server uses only the Python standard library. No package installation is needed.

## Explore

Switch between Tiered, Network, Upstream, and All views, or between 3D and 2D. Use the search bar and filters to find entities or relationships. Select a graph item to inspect it, and choose Details to focus on its connections. The right-hand panel shows graph information, evidence, and sources.

You can import another compatible graph JSON file from the Dataset menu. The bundled dataset loads automatically.

## Project files

- `index.html` — page layout and controls
- `assets/css/styles.css` — visual styles
- `assets/js/app.js` — graph rendering and user interactions
- `assets/js/demo-data.js` — generated dataset used for immediate loading
- `server.py` — local HTTP server and evidence retrieval
- `data/` — graph JSON, SQLite database, and source cache
- `tools/build_query_db.py` — rebuild the SQLite database from a graph export
- `tools/build_data_catalog.py` — update the list of available JSON datasets

To rebuild the database, run `python tools/build_query_db.py`. To refresh the dataset list, run `python tools/build_data_catalog.py`.

Evidence retrieval needs an internet connection. The app can still display the bundled graph without external access.
