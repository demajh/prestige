#!/usr/bin/env python3
"""Basic usage example for Prestige Python bindings.

This example demonstrates:
- Opening a store
- Basic put/get/delete operations
- Deduplication statistics
- Context manager usage
"""

import prestige
import tempfile
import shutil
from pathlib import Path


def main():
    # Create a temporary directory for the database
    db_path = Path(tempfile.mkdtemp()) / "example_db"
    print(f"Creating database at: {db_path}")

    try:
        # Open the store using context manager (recommended)
        with prestige.open(str(db_path)) as store:
            print("\n=== Basic Operations ===")

            # Put some values
            store.put("user:1", "Alice")
            store.put("user:2", "Bob")
            store.put("user:3", "Charlie")
            print("✓ Stored 3 user records")

            # Get a value
            value = store.get("user:1", decode=True)
            print(f"✓ Retrieved user:1 = {value}")

            # Check if key exists
            if "user:2" in store:
                print("✓ user:2 exists in store")

            # Delete a key
            store.delete("user:3")
            print("✓ Deleted user:3")

            print("\n=== Deduplication Demo ===")

            # Store the same content under many different keys
            duplicate_content = "This is duplicate content"
            for i in range(100):
                store.put(f"dup:{i}", duplicate_content)

            print(f"✓ Stored same content under 100 different keys")

            # Check deduplication stats
            total_keys = store.count_keys()
            unique_values = store.count_unique_values()
            print(f"✓ Total keys: {total_keys}")
            print(f"✓ Unique values: {unique_values}")
            print(f"✓ Deduplication ratio: {total_keys / unique_values:.1f}x")

            print("\n=== Health Statistics ===")

            # Get detailed health stats
            health = store.get_health()
            print(f"Total keys: {health['total_keys']}")
            print(f"Total unique objects: {health['total_objects']}")
            print(f"Storage used: {health['total_bytes'] / 1024:.1f} KB")
            print(f"Dedup ratio: {health['dedup_ratio']:.2f}x")

            print("\n=== List Keys ===")

            # List all user keys
            user_keys = store.list_keys(prefix="user:")
            print(f"User keys: {user_keys}")

            # List first 5 duplicate keys
            dup_keys = store.list_keys(prefix="dup:", limit=5)
            print(f"First 5 duplicate keys: {dup_keys}")

            print("\n=== Dict-like Interface ===")

            # You can also use dict-like syntax
            store["new_key"] = b"new value"
            value = store["new_key"]
            print(f"✓ Used dict syntax: {value}")

            # Get approximate count (fast)
            approx_count = len(store)
            print(f"✓ Approximate key count: {approx_count}")

            print("\n=== Binary Data ===")

            # Store binary data
            binary_data = bytes([0x00, 0x01, 0x02, 0xFF])
            store.put("binary_key", binary_data)
            retrieved = store.get("binary_key")
            print(f"✓ Stored and retrieved binary data: {retrieved.hex()}")

        # Store is automatically closed when exiting context manager
        print("\n✓ Store closed successfully")

        print("\n=== Persistence ===")

        # Reopen the same database to verify persistence
        with prestige.open(str(db_path)) as store:
            # Data should still be there
            value = store.get("user:1", decode=True)
            print(f"✓ Reopened database, user:1 = {value}")
            print(f"✓ Total keys persisted: {store.count_keys()}")

    finally:
        # Clean up
        if db_path.exists():
            shutil.rmtree(db_path.parent)
            print(f"\n✓ Cleaned up database at {db_path}")


if __name__ == "__main__":
    main()
