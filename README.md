# 🚕 Chicago Rideshare Operations Analysis

## 📌 Project Overview
This project is a comprehensive rideshare operations analysis built using **Python, SQL, and Tableau**. It provides data-driven insights into Chicago's Transportation Network Provider (TNP) trip activity, including:

- Demand patterns by hour and day of week
- Identification of high-volume pickup zones
- Congestion and trip-speed analysis across the day
- Fare and revenue breakdowns by zone and time
- Shared-ride adoption trends
- Weekday vs. weekend booking behavior

The goal of this dashboard is to help businesses, analysts, and city planners make informed operational decisions based on real rideshare trip data.

## 🚀 Features
- ✅ **Interactive Dashboard:** User-friendly interface with Date and Community Area filters.
- ✅ **Dynamic Visualizations:** Includes trend lines, bar charts, donut charts, and a zone-level breakdown table.
- ✅ **ETL Pipeline:** Data pulled live via the Socrata Open Data API, cleaned and transformed with Python/Pandas.
- ✅ **Data Cleaning & Validation:** Removes invalid records, deduplicates, handles missing zone data, and engineers derived fields (hour, day of week, trip speed).
- ✅ **SQL Analysis Layer:** Full set of analytical queries covering demand, fare/revenue, shared-ride behavior, zone comparisons, and trip efficiency.
- ✅ **Custom Insights:** Includes findings such as the highest-fare zone, peak congestion hour, and busiest pickup zone.

## 📂 Dataset Used
**Data Source:** [City of Chicago Data Portal — Transportation Network Providers (TNP) Trips (2025-)](https://data.cityofchicago.org/d/6dvr-xwnh), which aggregates rideshare trip activity across Chicago-licensed TNP operators. This is the City's current, continuously updated TNP dataset (covering 2025 onward); the June-August 2026 data used in this project falls within it.

**Note on scope:** this dataset does not include a provider-level identifier (i.e., it cannot distinguish which specific company operated a given trip). All findings in this project describe aggregate TNP/rideshare activity citywide, not any single provider.

**Access method:** Pulled directly via the Socrata Open Data API (not a static download). Raw CSV data is intentionally excluded from this repo (see `.gitignore`) — the notebook below can be re-run at any time to regenerate the full dataset live from the source.

**Period:** June – August 2026

**Sample size and methodology:** ~286,000 trips. The full 3-month dataset contains far more records than needed for this analysis, so a representative sample was taken rather than a full pull, to keep the dataset manageable in size while avoiding sampling bias. Specifically:
- A naive "pull the first N rows" approach was tested first and found to introduce **severe bias** — because the API returns rows in roughly chronological order by default, an unordered pull with a row limit captured only the first 1-2 calendar days of the window, skewing every downstream hour/day pattern.
- The final approach instead queried the API **once per hour, per day**, across the full 3-month range (92 days × 24 hours), pulling a small, fixed number of rows per hour-block. This ensures every hour of every day is represented proportionally, making hour-of-day and day-of-week comparisons valid.
- This was verified directly: the final sample shows an even distribution across all 24 hours (~11,900-11,950 trips each) and all 7 days of the week (~40,400-43,600 trips each), confirmed via `value_counts()` before proceeding to analysis.

**Key fields used:** trip start/end time, pickup/dropoff community area, trip miles, trip seconds, fare, tip, trip total, shared-trip flags.

## 🔧 Technologies Used
| Technology | Purpose |
|---|---|
| Python (Pandas) | Data extraction (API), cleaning, feature engineering |
| MySQL | Data storage & SQL analysis |
| SQL | Aggregations, subqueries, conditional logic |
| Tableau | Dashboard design & interactive visualization |

## 📊 Dashboard Insights & Business Recommendations

**1. Demand is heavily concentrated downtown**
- **Insight:** Near North Side alone generated **37,160 trips** — roughly double the next-highest zone (The Loop, 20,171) — with the top zones mapping cleanly onto Chicago's dense downtown, nightlife, and airport corridors.
- **Business implication:** Driver supply that isn't proactively positioned toward these zones likely sits idle elsewhere while demand goes underserved downtown, increasing rider wait times during peak windows.
- **Recommendation:** Use zone-level demand data to inform driver positioning incentives (e.g., surge-ahead notifications or repositioning bonuses) that nudge idle drivers toward Near North Side, The Loop, and Near West Side ahead of predictable peak periods.

**2. Average trip speed drops nearly 50% during evening rush hour**
- **Insight:** Average speed falls from a peak of **~32 mph around 5 AM** to a low of **~15.3 mph around 5 PM**, a clean congestion signature across the day.
- **Business implication:** Slower trips mean drivers complete fewer rides per hour during rush periods, reducing their effective earnings per hour even if fares stay flat — which can push drivers to avoid exactly the hours when rider demand is highest.
- **Recommendation:** Weight dynamic pricing or driver incentives more heavily during the 4-6 PM window specifically, to offset the reduced trips-per-hour drivers can complete and keep supply from thinning out during peak congestion.

**3. O'Hare drives the highest fares, but not the highest volume**
- **Insight:** Average fare at O'Hare (**$36.51**) is more than double the citywide busiest zone (Near North Side, $17.08), driven by trip distance — yet O'Hare's trip volume (18,028) is roughly half of Near North Side's.
- **Business implication:** O'Hare represents a high-value-per-trip zone that may be relatively under-served relative to its revenue potential, since driver attention naturally gravitates toward high-*volume* zones rather than high-*value* ones.
- **Recommendation:** Design a targeted driver incentive (e.g., guaranteed minimum earnings or priority matching) specifically for O'Hare pickups, to pull incremental driver supply toward a zone where each trip is disproportionately valuable.

**4. Shared-ride adoption remains low overall**
- **Insight:** Only ~3.24% of trips were shared, with modest peaks in early morning and late evening.
- **Business implication:** Low pooling adoption means more vehicle-miles driven per rider, which is both a cost-efficiency and sustainability gap.
- **Recommendation:** Test stronger pricing incentives or improved in-app visibility for shared-ride options, particularly during the hours where adoption already shows slight natural uptake, to validate whether incentive-sensitivity is the primary barrier.

| Metric | Value |
|---|---|
| Total Trips | 286,434 |
| Average Fare | $17.89 |
| Average Speed | 21.71 mph |
| Shared-Ride Rate | 3.24% |

## 🔗 Live Dashboard

[![Dashboard Preview](dashboard-overview.png)](https://public.tableau.com/app/profile/ayesha.ansari2317/viz/chicagorideshareOperationsAnalysis/ChicagoRideshareOperationsDashboard_?publish=yes)

**Click the image above to explore the interactive Tableau dashboard live.**

## 📁 Repository Contents
- [`chicago_rideshare_analytics.ipynb`](chicago_rideshare_analytics.ipynb) — Data extraction & cleaning (Python/Pandas)
- [`queries.sql`](queries.sql) — Full SQL analysis query set
- [`requirements.txt`](requirements.txt) — Python dependencies

## ▶️ How to Reproduce
1. Clone this repo.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Open the notebook in Jupyter or Google Colab and run all cells — this pulls fresh data directly from the Chicago Data Portal API using the hour-by-hour sampling method described above.
4. Load the resulting cleaned CSV into MySQL using the Table Data Import Wizard (or `LOAD DATA LOCAL INFILE`), then run `queries.sql` in MySQL Workbench to reproduce the analysis.
5. (Optional) Connect Tableau to the resulting table to rebuild the dashboard.

## 📄 License
This project is licensed under the MIT License — see [LICENSE](LICENSE) for details. The underlying dataset is sourced from the City of Chicago Data Portal and used under its open data terms.

---

Feel free to star ⭐ this repository if you found it useful! 🚀
