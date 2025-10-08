Purpose: portable, fast execution of the best baseline (geometric mean + seasonality). This image only runs the best method by default.

Build image:

```bash
docker build -t classicml -f docker/Dockerfile .
```

Run best baseline (Linux/macOS):

```bash
docker run --rm \
  -v "$(pwd)/data/raw:/app/data/raw" \
  -v "$(pwd)/submissions:/app/submissions" \
  classicml
```

Run best baseline (Windows PowerShell):

```powershell
docker run --rm `
  -v "${PWD}/data/raw:/app/data/raw" `
  -v "${PWD}/submissions:/app/submissions" `
  classicml
```

Output will be written to `/app/submissions/baseline_seasonality.csv` (mapped to `submissions/` on the host).

Advanced: you can set MODE to `geometric`, `simple`, or `ridge` to run other methods, but `baseline` is fastest and recommended.
