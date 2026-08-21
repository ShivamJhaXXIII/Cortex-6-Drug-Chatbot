from src.web_ingestion import fetch_official_source


url = input(
    "Enter official drug URL: "
).strip()


try:

    filepath = fetch_official_source(
        url
    )

    print()
    print("SUCCESS")
    print("Downloaded:", filepath)

except Exception as e:

    print()
    print("FAILED")
    print(e)