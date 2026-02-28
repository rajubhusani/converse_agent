#!/usr/bin/env python3
"""
Database Migration — Create/update tables from SQLAlchemy models.

Usage:
    # Local:
    python scripts/migrate_db.py

    # Inside Docker:
    docker compose -f docker-compose.prod.yml run --rm app python scripts/migrate_db.py

    # Check status only (no changes):
    python scripts/migrate_db.py --check
"""
import asyncio
import os
import sys
import argparse

# Ensure app root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def run_migration(check_only: bool = False):
    from config.settings import load_settings
    load_settings()

    from database.session import get_engine
    from database.models import Base

    engine = get_engine()

    if check_only:
        dialect = engine.dialect.name
        print(f"Database: {dialect}")
        print(f"URL: {str(engine.url).split('@')[-1] if '@' in str(engine.url) else str(engine.url)}")
        print(f"Tables defined: {', '.join(Base.metadata.tables.keys())}")

        async with engine.connect() as conn:
            from sqlalchemy import text

            # Database-specific table listing
            if dialect == "postgresql":
                result = await conn.execute(text(
                    "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
                ))
            elif dialect == "mysql":
                result = await conn.execute(text("SHOW TABLES"))
            else:  # sqlite
                result = await conn.execute(text(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ))

            existing = [row[0] for row in result.fetchall()]
            print(f"Tables existing: {', '.join(existing) or '(none)'}")

            missing = set(Base.metadata.tables.keys()) - set(existing)
            if missing:
                print(f"Tables MISSING: {', '.join(missing)}")
                print("Run without --check to create them.")
            else:
                print("All tables exist. ✓")
        return

    print("Running database migration...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Verify
    async with engine.connect() as conn:
        from sqlalchemy import text
        result = await conn.execute(text(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        ))
        tables = [row[0] for row in result.fetchall()]
        print(f"Tables created/verified: {', '.join(tables)}")

    await engine.dispose()
    print("Migration complete. ✓")


def main():
    parser = argparse.ArgumentParser(description="Database migration")
    parser.add_argument("--check", action="store_true", help="Check status only")
    args = parser.parse_args()

    asyncio.run(run_migration(check_only=args.check))


if __name__ == "__main__":
    main()
