# Supabase setup

HuggingArch uses Supabase for auth, profile storage, login analytics, the Kernel Bench lake lease, BYOK credentials, and user-contributed specs. `init.sql` is the whole story: a fresh deployment initializes from it, and the production database is already at that shape.

## Layout

```
supabase/
  init.sql                   — the complete schema baseline (folded through 0041,
                               then the restarted 0000–0026, on 2026-10-09)
  migrations/
    NNNN_<description>.sql   — schema changes during a cycle; folded into
                               init.sql once applied to production
                               (empty after the 2026-10-09 fold; the next
                               migration is 0027)
  README.md
```

`init.sql` is generated from the live, fully-migrated database
(`pg_dump --schema-only --no-owner -n public`, plus the header and the
not-empty guard). Migrations `0001`–`0041` were folded into it and deleted;
they live in git history. The restarted numbering (`0000`–`0026`) is folded
too:

- the Kernel Bench migrations: `init.sql` carries the lake lease (`0023`)
  and none of the Postgres measurement store `0026` dropped. The bench
  migrations that built that store (`0001`, `0004`, `0006`, `0007`,
  `0009`–`0019`, `0021a`–`0026`) were deleted with it — `0023`'s lease lives
  in `init.sql` (service role: SELECT on the table, EXECUTE on
  `bench_lake_claim`; the transfer is an operator's), `0026`'s drop is the
  baseline's shape;
- the spec-generation and credential migrations, folded on 2026-10-09 after
  checking that production had every one of them: `0000_credential_names`,
  `0002_spec_generation_engine_blocked`, `0003_pending_detail`,
  `0005_generation_log_record`, `0008_run_error_kind_agent`,
  `0020_review_log`.

The fold also left one copy of the dump in `init.sql`: the 0041 refold had
appended a second copy whose guard stopped an `ON_ERROR_STOP` run, so a
fresh install got the first, stale copy (no `agent_cli` / `agent_model` on
`spec_generation_logs`).

The dump assumes what a Supabase project provides: the roles `anon`,
`authenticated`, `service_role` and `supabase_admin` (grants and default
privileges name them) and the `auth` schema (`auth.users`, `auth.uid()`).
A plain Postgres needs stand-ins for those first (`tests/supabase/` has them).

## Initialize a new project

On a **new, empty Supabase project**, run the single baseline:

```bash
psql "$SUPABASE_DB_URL" -v ON_ERROR_STOP=1 -f supabase/init.sql
```

Or paste `supabase/init.sql` into **Supabase Studio → SQL Editor**. The script refuses to run when `public.profiles` already exists.

## Change the schema (existing project)

1. Add the next numbered migration under `supabase/migrations/` — the next
   number is `0027` (the restarted `0000`–`0026` are folded and live in git
   history; continuing the count keeps every file name unique) — and apply it:

   ```bash
   psql "$SUPABASE_DB_URL" -v ON_ERROR_STOP=1 -f supabase/migrations/0027_<description>.sql
   ```

2. Once production is at the new shape, refold the baseline so new projects
   get it in one file. *Replace* the body of `init.sql`, do not append:

   ```bash
   pg_dump --schema-only --no-owner -n public "$SUPABASE_DB_URL"
   ```

   `init.sql` is the header comment, the dump's `SET` lines, the not-empty
   guard (the `DO` block refusing a database with `public.profiles`), then
   the rest of the dump with `CREATE SCHEMA public` made `IF NOT EXISTS` and
   the dump's banner and `\restrict` / `\unrestrict` lines dropped. Before
   committing, build a database from `init.sql` alone with
   `ON_ERROR_STOP=1` and check its `pg_dump` matches the live one.

3. Delete the folded migration files again — the baseline plus git history are the record.

Never edit a migration that has been applied anywhere; a schema change always ships as a new file.

## Rolling out the engine-blocked spec records (former 0002)

Migration `0002_spec_generation_engine_blocked.sql` (now folded into
`init.sql`; production has it) added pending generation records
(`blocked_by_engine`, `needs_repair`) beside a user's valid ones. A backend
from before 0002 serves a user's newest record that is not `rejected`, so it
would serve a pending record as if it were a valid spec. Production deploys
roll (`flyctl deploy --strategy rolling`): old and new instances overlap, and
draining the old ones cannot be assumed. The new backend therefore writes
pending records only when `HA_RECORD_BLOCKED_SPECS=1`. Until then a blocked
run ends with its gaps on the run and the note that its spec was not saved.

1. The schema is in place (production applied 0002; a new project gets it
   from `init.sql`).
2. Deploy the new backend with `HA_RECORD_BLOCKED_SPECS` unset.
3. Check that no instance runs the old backend any more (`flyctl status`:
   every machine on the new release).
4. Enable the pending records:

   ```bash
   flyctl secrets set HA_RECORD_BLOCKED_SPECS=1
   ```

   This restarts the machines; they all run the new backend, so the overlap
   is new with new.

Rolling back past this backend: unset `HA_RECORD_BLOCKED_SPECS` first, then
delete the pending records an older backend would serve
(`DELETE FROM public.spec_generations WHERE status IN ('blocked_by_engine',
'needs_repair');`), then deploy the older release.

## Kernel Bench measurements

Measured kernel benchmarks are not stored in Postgres tables any more: they
live in a DuckLake (docs/measured_calibration.md §4) on the API
machine's volume — a catalog file and a directory of Parquet files — opened
by the API process from the environment:

```bash
BENCH_LAKE_CATALOG=/app/.cache/bench_lake/catalog.ducklake
BENCH_LAKE_DATA=/app/.cache/bench_lake/data/
BENCH_LAKE_BACKUP_DATA=s3://<bucket>/<prefix>    # backups leave the machine
BENCH_LAKE_BACKUP_S3_ENDPOINT=... BENCH_LAKE_BACKUP_S3_REGION=... BENCH_LAKE_BACKUP_S3_KEY=... BENCH_LAKE_BACKUP_S3_SECRET=...
```

Without the first two every `/api/bench/*` endpoint answers 503 naming the
missing setting. At startup the API creates the lake's tables (or restores
the lake from the backup, `HUGGINGARCH_BENCH_RESTORE`) and claims the lease
in `public.bench_lake_lease`. To seed the shared corpus from a directory of CSVs (owned by an admin):

```bash
uv run python scripts/seed_shared_corpus.py --owner <admin-user-uuid> --dir <dir of CSVs>
```

The Postgres bench objects (`user_benchmarks`, `user_benchmark_sets`,
`bench_*`) were dropped by `0026` after the one-time import into the lake; a
dump of them is in the backup bucket (`legacy-postgres-20261009/`). Only the
lease (`bench_lake_lease`, `bench_lake_claim`, `bench_lake_transfer`) is left
in Postgres.

## Test the functions on a real Postgres

Functions that decide on existing rows (the spec-generation record writes,
from the former `0002`) have behaviour tests in `tests/supabase/`. They build a disposable
local cluster, load `init.sql` and every migration, and call the functions
directly; they never connect to an existing database, and skip when no
Postgres server binaries are found:

```bash
HA_TEST_PG_BIN=/usr/lib/postgresql/16/bin python -m pytest tests/supabase
```
