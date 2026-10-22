# VEGA & SPARROW Defect Monitoring Dashboard

This version replaces the Plotly grid chart with 960 individually rendered unit tiles. The top-right tile is Unit 1; Unit 2 is directly underneath, and numbering follows the serpentine pattern from the provided layout. Hover over a tile to see X, Y, unit number, and reject count.

## Run on Windows
1. Extract this ZIP into a new folder.
2. Open a terminal in that folder.
3. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
4. Check the syntax:
   ```powershell
   python -m py_compile app.py
   ```
5. Launch:
   ```powershell
   python -m streamlit run app.py
   ```

## Data input
The app supports `.xlsx` and `.csv`. For the supplied UGP workbook it reads the first worksheet and recognizes `ProdDate`, `USMDevice`, `MESLotID`, `IFXLotNo`, `StripID`, `StripX`, `StripY`, `MachineRejectCode`, and `UserRejectCode`.

The color legend is unchanged: green = 0 rejects, pink = 1–4, orange = 5–14, red = 15 or more.
