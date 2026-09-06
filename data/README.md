# Data

## `processed/sp500_prices.csv.gz`

Wide panel of daily closing prices: 1,258 trading days (13 August 2012 to
11 August 2017) by 428 US large-cap tickers, gzipped CSV, 1.3 MB.

Produced by `scripts/01_build_dataset.py`, which:

1. pivots the raw long-format file into a date-by-ticker panel;
2. keeps only tickers with a complete price history over the window
   (503 raw tickers, 447 with full history);
3. interpolates isolated bad prints and **drops any ticker with an
   unexplained persistent jump** (19 dropped, 428 retained).

Step 3 is the interesting one — see the module docstring in
`src/robustcov/data.py` and the Data section of the top-level README for why
tickers are excluded rather than back-adjusted.

## Provenance

The raw file is `all_stocks_5yr.csv` from the
[orest-d/sp500exp](https://github.com/orest-d/sp500exp) repository, a mirror
of the widely redistributed Kaggle "S&P 500 stock data" set. Prices are
closes, not dividend-adjusted total returns.

The cleaned panel is vendored here so results reproduce without network
access. Check the upstream terms before any commercial use.

## Known limitations

- **Survivorship bias.** The universe is names in the index late in the sample
  that also have a complete price history, so companies that failed or were
  acquired during the window are absent.
- **No dividends**, so level returns are understated by roughly 2% a year.
  Immaterial for daily covariance estimation.
- **Split adjustment is incomplete upstream**, which is what step 3 handles.

## `processed/blue_chip_2006_2018.csv.gz`

Wide panel of daily closing prices: 3,020 trading days (3 January 2006 to
29 December 2017) by 23 blue-chip tickers, gzipped CSV, 171 KB. Built
specifically to span the NBER-dated December 2007 – June 2009 recession,
which the main panel above does not cover.

Produced by `scripts/01b_build_crisis_dataset.py`. The same cleaning function
(`clean_price_panel`) was run on it: all 23 tickers passed untouched — the
largest daily moves (GS, JPM, TRV, AXP, UNH, all clustered in
September–November 2008 and January–March 2009) were checked individually
and are genuine crisis-era moves, not corporate-action artifacts.

**Why only 23 names, not hundreds.** Requiring a *complete* daily history
back to January 2006 excludes anything that IPO'd, spun off, or had a data
gap afterward — AAPL, AMZN, GOOGL and MSFT are all excluded here purely for
short gaps elsewhere in their history, nothing to do with the crisis itself.
This is a real tradeoff: it is a second, independent dataset with the
distinct virtue of covering a genuine crash, but at `p=23` it cannot test
the high-dimensional (`p` comparable to `n`) regime at a realistic
estimation-window length — only by deliberately shrinking the window down to
match `p` (see `scripts/05_crisis_robustness.py` and the top-level README).

**Provenance.** From
[mayur-tikar/DJIA-30-Stock-Time-Series-Dataset](https://github.com/mayur-tikar/DJIA-30-Stock-Time-Series-Dataset),
itself redistributing data pulled via the AlphaVantage API. Prices are
closes, not dividend-adjusted total returns. Check the upstream terms before
any commercial use.
