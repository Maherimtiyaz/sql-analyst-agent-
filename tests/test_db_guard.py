"""The safety layer. These tests need no API key and should never be skipped."""
from __future__ import annotations

import hashlib

import pytest

from sql_agent.db import QueryError, ReadOnlyDB


def file_hash(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_select_works(db):
    r = db.query("SELECT COUNT(*) AS n FROM customers")
    assert r.columns == ["n"] and r.rows == [[300]] and not r.truncated


def test_cte_and_join_work(db):
    r = db.query("""
        WITH t AS (SELECT o.id, SUM(oi.quantity * oi.unit_price_cents) AS cents
                   FROM orders o JOIN order_items oi ON oi.order_id = o.id GROUP BY o.id)
        SELECT COUNT(*) FROM t""")
    assert r.rows[0][0] > 0


@pytest.mark.parametrize("sql", [
    "INSERT INTO customers (name, email, country, signup_date) VALUES ('x','x@x.io','X','2026-01-01')",
    "UPDATE orders SET status = 'completed'",
    "DELETE FROM orders",
    "DROP TABLE orders",
    "CREATE TABLE evil (id INTEGER)",
    "PRAGMA table_info(customers)",
    "ATTACH DATABASE ':memory:' AS other",
])
def test_writes_and_admin_statements_are_refused(db, db_path, sql):
    before = file_hash(db_path)
    with pytest.raises(QueryError):
        db.query(sql)
    assert file_hash(db_path) == before


def test_multiple_statements_are_refused(db):
    with pytest.raises(QueryError):
        db.query("SELECT 1; DELETE FROM orders")


def test_restricted_column_is_blocked(db):
    with pytest.raises(QueryError, match="customers.email"):
        db.query("SELECT email FROM customers WHERE id = 1")


def test_select_star_on_restricted_table_is_blocked_with_hint(db):
    with pytest.raises(QueryError, match="SELECT \\*"):
        db.query("SELECT * FROM customers LIMIT 1")


def test_restricted_column_cannot_be_reached_indirectly(db):
    for sql in ["SELECT COUNT(DISTINCT email) FROM customers",
                "SELECT name FROM customers WHERE email LIKE 'user1@%'",
                "SELECT c.id FROM customers c ORDER BY c.email LIMIT 1"]:
        with pytest.raises(QueryError):
            db.query(sql)


def test_other_customer_columns_are_fine(db):
    assert db.query("SELECT name, country FROM customers WHERE id = 1").rows


def test_slow_query_hits_timeout(db_path):
    slow = ReadOnlyDB(db_path, timeout_s=0.2)
    with pytest.raises(QueryError, match="time limit"):
        slow.query("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) SELECT COUNT(*) FROM n")


def test_row_cap_sets_truncated(db_path):
    small = ReadOnlyDB(db_path, max_rows=5)
    r = small.query("SELECT id FROM orders")
    assert len(r.rows) == 5 and r.truncated


def test_describe_table_hides_restricted_values(db):
    d = db.describe_table("customers")
    email = next(c for c in d["columns"] if c["name"] == "email")
    assert email.get("restricted") is True
    assert "email" not in d["sample_rows"]["columns"]
    assert "@" not in str(d["sample_rows"]["rows"])


def test_describe_unknown_table_lists_real_tables(db):
    with pytest.raises(QueryError, match="order_items"):
        db.describe_table("order")
