# Data Pipeline

Run `python pipeline.py` from this directory or `python data_pipeline/pipeline.py` from the repository root. The script scrapes five catalogue pages (100 books), which naturally spans many categories, and fetches each product page to capture its category.

Cleaning decisions: price symbols are removed and prices parsed as floats; text ratings map from One--Five to 1--5; availability becomes a boolean. Unexpected numeric values are median-imputed, while records without title/category are dropped because neither can be reliably inferred. `price_inr = price_gbp * 105.50`, using the required fixed project baseline.

`catalog.db` contains normalized `categories` and `books` tables with a foreign key. Five query result CSVs are generated, including a JOIN. The script reads SQL results with `pd.read_sql` and asserts that the JOIN is identical to an in-memory `pd.merge` result.
