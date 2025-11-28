"""Fix the corrupted test_cases.py file by replacing literal \\n with actual newlines."""

# Read the corrupted file
with open('benchmarks/test_cases.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace literal \n with actual newlines
content = content.replace('\\n', '\n')

# Write back
with open('benchmarks/test_cases.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed corrupted file")
