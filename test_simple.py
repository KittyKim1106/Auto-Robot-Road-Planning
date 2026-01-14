try:
    from app import app
    print("Import success")
except Exception as e:
    print(f"Import failed: {e}")
