"""Script to reset the database by dropping all tables and recreating them.

This script will:
1. Drop all existing tables
2. Recreate the tables with the current schema
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from app.db.base import BaseModel
    from app.db.session import engine
except ImportError:
    sys.exit(1)


def reset_database() -> None:
    """Reset the database by dropping and recreating all tables.

    Args:
        populate: Whether to populate the database with sample data after reset

    """
    try:
        BaseModel.metadata.drop_all(bind=engine)
        BaseModel.metadata.create_all(bind=engine)

    except Exception:
        sys.exit(1)


def main() -> None:
    """Main function to handle command line arguments and execute reset."""

    if not os.getenv("SKIP_CONFIRMATION"):
        response = input("This will DELETE ALL DATA in the database. Continue? (y/N): ")
        if response.lower() != "y":
            return

    reset_database()


if __name__ == "__main__":
    main()
