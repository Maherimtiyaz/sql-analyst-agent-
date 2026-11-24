"""Create data/shop.db: a small, deterministic e-commerce database.

Same seed -> same data -> eval answers stay stable.   Run:  python scripts/seed.py
"""
from __future__ import annotations

import os
import random
import sqlite3
import sys
from datetime import datetime, timedelta

PATH = "data/shop.db"

SCHEMA = """
CREATE TABLE customers (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  email TEXT NOT NULL UNIQUE,
  country TEXT NOT NULL,
  signup_date TEXT NOT NULL
);
CREATE TABLE products (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  category TEXT NOT NULL,
  price_cents INTEGER NOT NULL
);
CREATE TABLE orders (
  id INTEGER PRIMARY KEY,
  customer_id INTEGER NOT NULL REFERENCES customers(id),
  status TEXT NOT NULL CHECK (status IN ('completed','pending','cancelled','refunded')),
  created_at TEXT NOT NULL
);
CREATE TABLE order_items (
  id INTEGER PRIMARY KEY,
  order_id INTEGER NOT NULL REFERENCES orders(id),
  product_id INTEGER NOT NULL REFERENCES products(id),
  quantity INTEGER NOT NULL CHECK (quantity > 0),
  unit_price_cents INTEGER NOT NULL
);
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_created ON orders(created_at);
CREATE INDEX idx_items_order ON order_items(order_id);
"""

FIRST = ["Aarav", "Mahek", "Liam", "Sofia", "Kenji", "Amara", "Lucas", "Priya", "Noah", "Zara",
         "Mateo", "Aisha", "Ethan", "Yuki", "Omar", "Elena", "Ravi", "Chloe", "Diego", "Fatima"]
LAST = ["Shah", "Khan", "Smith", "Rossi", "Tanaka", "Okafor", "Silva", "Patel", "Brown", "Ali",
        "Garcia", "Haddad", "Nguyen", "Sato", "Mehta", "Novak", "Iyer", "Martin", "Lopez", "Rahman"]
COUNTRIES = [("India", 30), ("United States", 25), ("United Kingdom", 12), ("Germany", 10),
             ("Brazil", 8), ("Japan", 8), ("Nigeria", 7)]
PRODUCTS = [
    ("Mechanical Keyboard", "Electronics", 8999), ("Wireless Mouse", "Electronics", 2999),
    ("USB-C Hub", "Electronics", 4599), ("27in Monitor", "Electronics", 24999),
    ("Noise-Cancelling Headphones", "Electronics", 17999), ("Webcam 1080p", "Electronics", 5999),
    ("Standing Desk", "Furniture", 39999), ("Ergonomic Chair", "Furniture", 29999),
    ("Monitor Arm", "Furniture", 7999), ("Desk Lamp", "Furniture", 3499),
    ("Clean Code", "Books", 3299), ("Designing Data-Intensive Applications", "Books", 4499),
    ("The Pragmatic Programmer", "Books", 3999), ("Database Internals", "Books", 4799),
    ("Cold Brew Coffee Beans", "Grocery", 1899), ("Green Tea Sampler", "Grocery", 1499),
    ("Protein Bars (12)", "Grocery", 2399), ("Hoodie", "Apparel", 4999),
    ("T-Shirt", "Apparel", 1999), ("Laptop Backpack", "Apparel", 6999),
]
STATUSES = [("completed", 80), ("pending", 7), ("cancelled", 8), ("refunded", 5)]
START, END = datetime(2025, 10, 1), datetime(2026, 9, 30, 23, 59, 59)


def pick(rng: random.Random, weighted: list[tuple[str, int]]) -> str:
    return rng.choices([w[0] for w in weighted], weights=[w[1] for w in weighted])[0]


def main(path: str = PATH) -> None:
    rng = random.Random(42)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)

    customers = []
    for cid in range(1, 301):
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        signup = START - timedelta(days=rng.randint(0, 400)) + timedelta(days=rng.randint(0, 300))
        customers.append((cid, name, f"user{cid}@example.com", pick(rng, COUNTRIES), signup.date().isoformat()))
    conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", customers)
    conn.executemany("INSERT INTO products VALUES (?,?,?,?)", [(i + 1, *p) for i, p in enumerate(PRODUCTS)])

    # ~15% of customers never order; the rest order with a long-tail distribution.
    buyers = [c[0] for c in customers if rng.random() > 0.15]
    weights = [rng.paretovariate(1.5) for _ in buyers]
    span = int((END - START).total_seconds())
    order_id = item_id = 0
    orders, items = [], []
    for _ in range(2500):
        order_id += 1
        created = START + timedelta(seconds=rng.randint(0, span))
        orders.append((order_id, rng.choices(buyers, weights=weights)[0], pick(rng, STATUSES),
                       created.strftime("%Y-%m-%dT%H:%M:%SZ")))
        for pid in rng.sample(range(1, len(PRODUCTS) + 1), k=rng.choice([1, 1, 1, 2, 2, 3])):
            item_id += 1
            price = PRODUCTS[pid - 1][2]
            if rng.random() < 0.2:  # discounted sale price
                price = int(price * rng.choice([0.8, 0.85, 0.9]))
            items.append((item_id, order_id, pid, rng.choice([1, 1, 1, 2, 3]), price))
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?)", orders)
    conn.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", items)
    conn.commit()
    conn.close()
    print(f"Wrote {path}: {len(customers)} customers, {len(PRODUCTS)} products, {len(orders)} orders, {len(items)} items")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else PATH)
