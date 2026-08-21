import easyocr


reader = easyocr.Reader(
    ["en"],
    gpu=False
)


image_path = "finasteride.jpeg"


results = reader.readtext(
    image_path
)


print("\nOCR RESULT")
print("=" * 60)


for bbox, text, confidence in results:

    print(
        f"{confidence:.2f}  {text}"
    )