"""Fix syntax errors caused by comma placement in test_cases.py"""
import re

# Read the file
with open('benchmarks/test_cases.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix pattern: move comma before comment
# Pattern: "...",  # comment,  -> "...",  # comment
content = re.sub(
    r'(\"[^\"]+\")  (# [^,\n]+),',
    r'\1,  \2',
    content
)

# Write back
with open('benchmarks/test_cases.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed syntax errors - moved commas before comments")
