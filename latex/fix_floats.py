import re

filename = "DP_Nela_Michalkova.tex"

with open(filename, "r", encoding="utf-8") as f:
    content = f.read()

# Replace \begin{figure}[...] and \begin{figure} with \begin{figure}[H]
new_content = re.sub(r'\\begin\{figure\}(?:\[.*?\])?', r'\\begin{figure}[H]', content)

# Replace \begin{table}[...] and \begin{table} with \begin{table}[H]
new_content = re.sub(r'\\begin\{table\}(?:\[.*?\])?', r'\\begin{table}[H]', new_content)

with open(filename, "w", encoding="utf-8") as f:
    f.write(new_content)

print("Floats fixed.")
