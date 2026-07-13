import re

filename = "DP_Nela_Michalkova.tex"

with open(filename, "r", encoding="utf-8") as f:
    content = f.read()

# Pattern for Table 1
table1_pattern = r"(\\begin\{table\}\[H\].*?\\label\{tab:exp1_rcci\}.*?\\end\{table\}\n)"
match1 = re.search(table1_pattern, content, flags=re.DOTALL)
if match1:
    table1_str = match1.group(1)
    # Remove from original
    content = content.replace(table1_str, "")
    # Insert after fig:exp1_r14
    fig1_end_pattern = r"(\\label\{fig:exp1_r14\}\n\\end\{figure\}\n)"
    content = re.sub(fig1_end_pattern, r"\1" + "\n" + table1_str.replace("\\", "\\\\") + "\n", content)

# Pattern for Table 2
table2_pattern = r"(\\begin\{table\}\[H\].*?\\label\{tab:exp2_mcoeff\}.*?\\end\{table\}\n)"
match2 = re.search(table2_pattern, content, flags=re.DOTALL)
if match2:
    table2_str = match2.group(1)
    # Remove from original
    content = content.replace(table2_str, "")
    # Insert after fig:exp2_m22
    fig2_end_pattern = r"(\\label\{fig:exp2_m22\}\n\\end\{figure\}\n)"
    content = re.sub(fig2_end_pattern, r"\1" + "\n" + table2_str.replace("\\", "\\\\") + "\n", content)


# Also we should change the wording in the text:
# "Následující tabulka a sada grafů" -> "Následující sada grafů a tabulka"
content = content.replace("Následující tabulka a sada grafů ilustrují", "Následující sada grafů a tabulka ilustrují")

with open(filename, "w", encoding="utf-8") as f:
    f.write(content)

print("Tables moved.")
