import re

# Extract our text from the text file
with open("text/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Split it into words taking weird simbold into consideration and then strip all the blank spaces from the result
preprocess = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocess = [item.strip() for item in preprocess if item.strip()]
print(preprocess[:90])
print( len(preprocess))