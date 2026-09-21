"""Scrape, clean, normalize, query and validate Books to Scrape data."""
from __future__ import annotations
import re, sqlite3
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).parent
DB = ROOT / "catalog.db"
RATE = 105.50
RATINGS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

def scrape_books() -> pd.DataFrame:
    rows = []
    for page in range(1, 6):
        url = "https://books.toscrape.com/catalogue/page-%d.html" % page
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for card in soup.select("article.product_pod"):
            title = card.h3.a["title"].strip()
            price = card.select_one(".price_color").get_text(strip=True)
            rating = card.select_one("p.star-rating")["class"][-1]
            availability = card.select_one(".availability").get_text(" ", strip=True)
            # Product pages carry the authoritative category; this keeps page scraping automatic.
            detail = requests.get("https://books.toscrape.com/catalogue/" + card.h3.a["href"], timeout=30)
            detail.raise_for_status()
            category = BeautifulSoup(detail.text, "html.parser").select("ul.breadcrumb li")[-2].get_text(strip=True)
            rows.append(dict(title=title, price_raw=price, star_rating=rating, availability_raw=availability, category=category))
    return pd.DataFrame(rows)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df["price_gbp"] = pd.to_numeric(df.price_raw.str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
    df["rating"] = df.star_rating.map(RATINGS)
    df["in_stock"] = df.availability_raw.str.contains("In stock", case=False, na=False)
    # Numeric parse errors use median imputation; rows missing essential text/category are dropped.
    for col in ["price_gbp", "rating"]:
        df[col] = df[col].fillna(df[col].median())
    df = df.dropna(subset=["title", "category"]).copy()
    df["rating"] = df.rating.astype(int)
    df["in_stock"] = df.in_stock.astype(bool)
    df["price_inr"] = (df.price_gbp * RATE).round(2)
    return df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]]

def load_database(df: pd.DataFrame) -> None:
    with sqlite3.connect(DB) as con:
        con.execute("PRAGMA foreign_keys = ON")
        con.executescript("""
        DROP TABLE IF EXISTS books; DROP TABLE IF EXISTS categories;
        CREATE TABLE categories(category_id INTEGER PRIMARY KEY, category_name TEXT UNIQUE NOT NULL);
        CREATE TABLE books(book_id INTEGER PRIMARY KEY, title TEXT NOT NULL, price_gbp REAL NOT NULL,
          price_inr REAL NOT NULL, rating INTEGER NOT NULL, in_stock INTEGER NOT NULL,
          category_id INTEGER NOT NULL REFERENCES categories(category_id));
        """)
        categories = pd.DataFrame({"category_name": sorted(df.category.unique())})
        categories.to_sql("categories", con, if_exists="append", index=False)
        lookup = pd.read_sql("SELECT * FROM categories", con)
        df.merge(lookup, left_on="category", right_on="category_name").drop(columns=["category", "category_name"]).to_sql("books", con, if_exists="append", index=False)

def run_queries() -> None:
    queries = {
        "01_where": "SELECT title, price_gbp FROM books WHERE in_stock = 1 LIMIT 5;",
        "02_order_limit": "SELECT title, price_inr FROM books ORDER BY price_inr DESC LIMIT 10;",
        "03_distinct": "SELECT DISTINCT rating FROM books ORDER BY rating;",
        "04_between": "SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 20 AND 30 LIMIT 10;",
        "05_join": "SELECT c.category_name, b.title, b.rating, b.price_inr FROM books b JOIN categories c ON b.category_id=c.category_id ORDER BY b.rating DESC, b.price_inr DESC LIMIT 10;",
    }
    with sqlite3.connect(DB) as con:
        for name, sql in queries.items():
            result = pd.read_sql(sql, con)
            (ROOT / f"{name}.csv").write_text(result.to_csv(index=False), encoding="utf-8")
            print(f"\n{name}\n{result.to_string(index=False)}")
        sql_join = pd.read_sql(queries["05_join"], con)
        books, cats = pd.read_sql("SELECT * FROM books", con), pd.read_sql("SELECT * FROM categories", con)
        pandas_join = books.merge(cats, on="category_id")[["category_name", "title", "rating", "price_inr"]].sort_values(["rating", "price_inr"], ascending=[False, False]).head(10).reset_index(drop=True)
        assert sql_join.reset_index(drop=True).equals(pandas_join), "JOIN comparison failed"
        print("\nSQL JOIN and pandas merge outputs match.")

if __name__ == "__main__":
    cleaned = clean(scrape_books())
    assert len(cleaned) >= 60
    cleaned.to_csv(ROOT / "cleaned_books.csv", index=False)
    load_database(cleaned)
    run_queries()
