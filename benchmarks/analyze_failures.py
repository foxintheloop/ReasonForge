"""Quick analysis script for benchmark failures."""
import json
from collections import Counter

# Load results
with open('benchmarks/results/benchmark_results_20251122_195636.json') as f:
    data = json.load(f)

failures = [r for r in data['detailed_results'] if not r['correct']]
successes = [r for r in data['detailed_results'] if r['correct']]

print(f'Total tests: {len(data["detailed_results"])}')
print(f'Passed: {len(successes)}')
print(f'Failed: {len(failures)}')
print(f'Accuracy: {len(successes)/len(data["detailed_results"])*100:.2f}%')
print('\nTop 15 failure reasons:')

reasons = Counter(r['explanation'][:80] for r in failures)
for reason, count in reasons.most_common(15):
    print(f'  {count:3d} - {reason}')

print('\nFailed test IDs:')
for r in failures[:20]:
    print(f"  {r['test_id']}: {r['explanation'][:60]}")
