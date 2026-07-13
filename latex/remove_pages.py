import re

filename = "DP_Nela_Michalkova.tex"

with open(filename, "r", encoding="utf-8") as f:
    content = f.read()

# Replace \cite[anything]{key} with \cite{key}
# Using non-greedy match .*? inside the brackets
new_content = re.sub(r'\\cite\[.*?\]\{', r'\\cite{', content)

with open(filename, "w", encoding="utf-8") as f:
    f.write(new_content)

print("Citations updated.")
