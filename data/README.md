# Data

Run `python src/download_data.py`. It calls the public openFDA endpoint
https://api.fda.gov/device/event.json filtered to manufacturer "ALCON".
Requires a free openFDA API key (https://open.fda.gov/apis/authentication/), passed with `--api-key` or the `OPENFDA_API_KEY` environment variable.
Output: `data/alcon_events.csv`.

Public U.S. government data. MAUDE reports are unverified submissions; see openFDA's disclaimer.
