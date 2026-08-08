"""
One-time setup: store the Lakebase connection URL in a Databricks secret scope.

Run from a Databricks notebook (`%sh python setup_secrets.py`) or locally with
the Databricks CLI configured. getpass keeps the value out of shell history and
off disk.

Note there is no API-key prompt here: the National Weather Service API is open
and unauthenticated, so the only secret this app needs is the database URL.
The NWS User-Agent string is configuration, not a credential, and lives in
app.yaml.

Usage:
    python setup_secrets.py
"""
import getpass

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace

w = WorkspaceClient()

# Uncomment on first run, if the scope doesn't exist yet.
# w.secrets.create_scope(scope="database")

w.secrets.put_secret(
    scope="database",
    key="lakebase-url",
    string_value=getpass.getpass("Paste your Lakebase URL: "),
)

w.secrets.put_acl(
    scope="database",
    principal="users",
    permission=workspace.AclPermission.READ,
)

print("Stored database/lakebase-url")
