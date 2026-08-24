import json
from collections import defaultdict

# Load JSON data
with open('data/detections.json', 'r') as f:
    pothole_data = json.load(f)

# Group entries by 'id'
id_groups = defaultdict(list)
for entry in pothole_data:
    id_groups[entry['id']].append(entry)

# Select the middle entry for each id
selected_entries = []
for pothole_id, entries in id_groups.items():
    middle_index = len(entries) // 2
    selected_entries.append(entries[middle_index])

# Now calculate total volume
total_volume = sum(entry['volume'] for entry in selected_entries)  # in cubic meters

# Print results
print(f"Total Pothole Volume to be filled: {total_volume:.3f} cubic meters")