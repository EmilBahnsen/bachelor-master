#!/usr/bin/python

import matplotlib.pyplot as plt
import sys

# Get the values into x, y1, and y2.
x = []
y1 = []
y2 = []
for line in sys.stdin:
  vals = line.split(',')
  x.append(float(vals[0]))
  y1.append(float(vals[1]))
  y2.append(float(vals[2]))

# Plot y1 vs x in blue on the left vertical axis.
plt.xlabel("x")
plt.ylabel("Blue", color="b")
plt.tick_params(axis="y", labelcolor="b")
plt.plot(x, y1, "b-", linewidth=2)

# Plot y2 vs x in red on the right vertical axis.
plt.twinx()
plt.ylabel("Red", color="r")
plt.tick_params(axis="y", labelcolor="r")
plt.plot(x, y2, "r-", linewidth=2)

plt.savefig("Two axes.png", dpi=75, format="png")
plt.close()

