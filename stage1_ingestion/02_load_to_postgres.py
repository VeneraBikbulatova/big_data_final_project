#!/usr/bin/env python3
import os
import sys
import time
import psycopg2

PG_HOST = os.environ['PG_HOST']
PG_PORT = os.environ['PG_PORT']
PG_DB = os.environ['PG_DB']
PG_USER = os.environ['PG_USER']
PG_PASSWORD = os.environ['PG_PASSWORD']
PG_TABLE = os.environ['PG_TABLE']
RAW_DATA_DIR = os.environ['RAW_DATA_DIR']
DATA_FILE = os.environ['DATA_FILE']
PROJECT_ROOT = os.environ['PROJECT_ROOT']

CSV_PATH = os.path.join(RAW_DATA_DIR, DATA_FILE)
SCHEMA_FILE = os.path.join(PROJECT_ROOT, 'config', 'postgres_schema.sql')


def _connect(dbname=None):
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT,
        dbname=dbname or PG_DB,
        user=PG_USER, password=PG_PASSWORD,
    )


def apply_schema():
    print(f"[INFO] Applying schema from {SCHEMA_FILE} ...")
    with open(SCHEMA_FILE, 'r') as f:
        ddl = f.read()
    conn = _connect()
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()
    conn.close()


def bulk_load_csv():
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] CSV file not found: {CSV_PATH}")
        sys.exit(1)

    size_gb = os.path.getsize(CSV_PATH) / 1e9
    print(f"[INFO] Loading {CSV_PATH}  ({size_gb:.2f} GB)")

    columns = ("event_time", "event_type", "product_id", "category_id",
               "category_code", "brand", "price", "user_id", "user_session")
    copy_sql = (
        f"COPY {PG_TABLE} ({', '.join(columns)}) "
        f"FROM STDIN WITH (FORMAT CSV, HEADER TRUE)"
    )

    conn = _connect()
    t0 = time.time()
    with conn.cursor() as cur, open(CSV_PATH, 'r') as f:
        cur.copy_expert(copy_sql, f)
    conn.commit()
    elapsed = time.time() - t0

    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {PG_TABLE}")
        n = cur.fetchone()[0]
    conn.close()

    print(f"[INFO] Loaded {n:,} rows in {elapsed:.1f}s "
          f"(~{n/elapsed:,.0f} rows/sec).")


def main():
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = %s", (PG_TABLE,))
    table_exists = cur.fetchone()[0] > 0
    
    if table_exists:
        cur.execute(f"SELECT COUNT(*) FROM {PG_TABLE}")
        count = cur.fetchone()[0]
        if count > 0:
            print(f"[INFO] Table {PG_TABLE} already has {count} rows. Skipping load.")
            conn.close()
            return
    conn.close()
    
    ensure_database_exists()
    apply_schema()
    bulk_load_csv()


if __name__ == "__main__":
    main()
