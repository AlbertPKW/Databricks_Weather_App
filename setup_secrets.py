"""
One-time setup script for WEATHER project: stores the student_three 
Lakebase connection URL.

Note: The National Weather Service API is open and unauthenticated, 
so no API key is needed.

Usage:
    python setup_secrets.py
"""
import getpass

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace

w = WorkspaceClient()

# Uncomment on first run if the scope doesn't exist yet
# w.secrets.create_scope(scope="database")

w.secrets.put_secret(
    scope="database",
    key="lakebase-url-weather",
    string_value=getpass.getpass("Paste student_three Lakebase URL: "),
)
print("✅ Stored as 'database/lakebase-url-weather'")

w.secrets.put_acl(
    scope="database",
    principal="users",
    permission=workspace.AclPermission.READ,
)
