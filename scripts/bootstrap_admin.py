"""Bootstrap the first admin user.

After applying supabase/{schema,functions,rls}.sql and signing up the
account that should be admin (via /auth/login.html), run:

    ADMIN_BOOTSTRAP_EMAIL=you@example.com python3 scripts/bootstrap_admin.py

This script uses the service-role key (bypasses RLS) to:
  1. Look up auth.users for the given email.
  2. Insert a row into public.admins with the user_id.

It does not depend on the temporary "anyone can INSERT" policy from
0003; migration 0028 drops that policy.

Idempotent: re-running with the same email is a no-op.
"""
from __future__ import annotations

import os
import sys

import httpx


def _required_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"error: {name} is not set", file=sys.stderr)
        sys.exit(2)
    return val


def main() -> int:
    url = _required_env("SUPABASE_URL").rstrip("/")
    service_key = _required_env("SUPABASE_SERVICE_ROLE_KEY")
    email = _required_env("ADMIN_BOOTSTRAP_EMAIL").lower().strip()

    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
    }

    # 1. Look up the auth.users row by email via the Auth Admin API.
    #    The /auth/v1/admin/users endpoint accepts ?email=<exact match>.
    with httpx.Client(timeout=10.0) as client:
        resp = client.get(
            f"{url}/auth/v1/admin/users",
            params={"email": email},
            headers=headers,
        )
        if resp.status_code != 200:
            print(f"auth lookup failed: {resp.status_code} {resp.text}", file=sys.stderr)
            return 1
        data = resp.json()
        users = data.get("users") if isinstance(data, dict) else data
        if not users:
            print(
                f"error: no auth.users row for {email!r}. "
                "Sign up first at /auth/login.html, verify the email, then re-run.",
                file=sys.stderr,
            )
            return 1
        user_id = users[0]["id"]
        print(f"found user_id={user_id} for {email}")

        # 2. Insert into public.admins. ON CONFLICT (user_id) DO NOTHING
        #    via the Prefer header so re-running is idempotent.
        ins = client.post(
            f"{url}/rest/v1/admins",
            headers={**headers, "Prefer": "return=representation,resolution=ignore-duplicates"},
            json={"user_id": user_id, "created_by": user_id},
        )
        if ins.status_code not in (200, 201, 409):
            print(f"insert failed: {ins.status_code} {ins.text}", file=sys.stderr)
            return 1
        if ins.status_code == 409:
            print(f"{email} is already an admin (no-op).")
        else:
            print(f"granted admin to {email}.")

    print()
    print("Next steps:")
    print("  Sign in at /auth/login.html, then visit /admin/ to see the dashboard.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
