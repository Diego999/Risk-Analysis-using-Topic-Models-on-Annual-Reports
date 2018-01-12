# Crawling-Edgar-Data-Annual-Report-10-k-From-U.S.-Securities-and-Exchange-Commission

Download annual report 10-k of companies through U.S. Securities and Exchange Commission from first quarter 1993 to last quarter of the current year.
Analyze Risk Factors section using topic modeling.
The code is delivered as it is. This project is not maintained.
You can find the paper [here](paper.pdf). 

# Requirements
It relies on Python 3.5. Due to a bug somewhere in Python & Gensim with multiprocessing, don't forget to run `export OMP_NUM_THREADS=1 && python3 myfile.py` for the step 4.

# Description of the dataset

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
