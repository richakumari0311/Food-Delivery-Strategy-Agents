# Migrating local Docker Postgres -> Supabase

Prerequisite: pg_dump / pg_restore installed locally.
  macOS: brew install libpq && brew link --force libpq

## 1. Dump the local Docker database

docker exec food_delivery_db pg_dump -U app -d food_delivery \
  --no-owner --no-privileges -F c -f /tmp/backup.dump
docker cp food_delivery_db:/tmp/backup.dump ./backup.dump

## 2. Confirm the dump actually has your tables (sanity check before restoring)

pg_restore --list ./backup.dump | grep -E "orders|user_features|reviews|investor_pdf_chunks"

You should see 4 lines, one per table. If this is empty, the dump itself
is empty and step 1 needs troubleshooting before going further.

## 3. Restore into Supabase

Get the connection string from Supabase dashboard > Settings > Database
> Connection string > URI. Use the Session pooler entry (port 5432), not
the Transaction pooler (port 6543).

pg_restore --no-owner --no-privileges --if-exists --clean \
  -d "postgresql://postgres:[YOUR-PASSWORD]@[YOUR-HOST]:5432/postgres" \
  ./backup.dump

Watch the output. pg_restore continues past individual errors by default
rather than stopping, so skim for lines containing "ERROR" - especially
around "CREATE EXTENSION vector", which may conflict with Supabase's
already-enabled extension. That specific error is safe to ignore if it
appears; everything else is worth reading closely.

## 4. Verify

python 06b_sanity_checks.py

Should return the same row counts (orders: 50000, user_features: 4000,
reviews: 4000, investor_pdf_chunks: 4021) as your original local run did.