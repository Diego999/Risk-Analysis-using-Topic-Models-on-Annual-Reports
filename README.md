# Crawling-Edgar-Data-Annual-Report-10-k-From-U.S.-Securities-and-Exchange-Commission
Download annual report 10-k of companies through U.S. Securities and Exchange Commission from first quarter 1993 to last quarter of the current year.

# Requirements
It relies on Python 3.5.

# Description

The script will create the followings folders:
- `data`
- `data/gz`: which contains the compressed version of company indices from EDGAR, grouped by quarter and year.
- `data/pd`: which contains the Pandas DataFrame version of company indices, grouped by quarter and year. Also includes a Pandas DataFrame merging all dataframes.
- `data/ar`: the annual reports
    - `data/ar/CIK/[names/*.txt]`: `names` contain all the company names associated with the CIK.
    - `txt` files are the annual reports.

At the end, the overall size is around 750 GB.

# Issues/Pull Requests/Feedbacks

Don't hesitate to contact for any feedback or create issues/pull requests.
