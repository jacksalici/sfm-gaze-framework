import pandas as pd
import pathlib, os
import toml

config = toml.load("config.toml")

csv = os.path.join(config["gaze_estimation"]["gaze_3d_output_folder"], "6_1_1.csv")

df = pd.read_csv(csv)

distance_min_stats = df["distance_min"].describe()
print(distance_min_stats)

import matplotlib.pyplot as plt
dark = '#546675'
light = '#E7ECF0'
red = '#F37983'
redish = '#FFD5C1'

fig, ax = plt.subplots()

violin_parts = ax.violinplot(df["distance_min"], vert=True, showmeans=True, showmedians=False, showextrema=True)
ax.set_facecolor(light)
plt.title('min distance violin plot', color=dark)
ax.set_xticks([1])
ax.set_xticklabels(['6_1_1'], color=dark)
ax.tick_params(axis='x', colors=dark)
ax.tick_params(axis='y', colors=dark)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

for partname in ('cbars','cmins','cmaxes','cmeans'):
    vp = violin_parts[partname]
    vp.set_edgecolor(red)
    vp.set_linewidth(2)

# Make the violin body blue with a red border:
for vp in violin_parts['bodies']:
    vp.set_facecolor(redish)
    vp.set_alpha(1)


ax.grid(True, which='both', linestyle='-', linewidth=0.5, color=dark)

plt.show()
