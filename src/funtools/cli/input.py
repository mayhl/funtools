from typing import Optional

import typer

app = typer.Typer(help="Awesome CLI user manager.")


@app.command()
def create(username: str):
    """Create a new user with USERNAME."""
    print(f"Creating user: {username}")


@app.command()
def delete(
    username: str, force: bool = typer.Option(False, "--force", help="Force deletion")
):
    """Delete a user with USERNAME."""
    if force:
        print(f"Deleting user: {username}")
    else:
        print("Operation cancelled")
