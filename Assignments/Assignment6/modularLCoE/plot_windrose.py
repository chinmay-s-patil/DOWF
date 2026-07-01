#!/usr/bin/env python
"""Standalone script to plot and save the wind rose."""

import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wind_data import create_wind_rose


def main():
    wind_rose = create_wind_rose()
    fig = wind_rose.plot()
    plt.savefig("wind_rose.png", dpi=300, bbox_inches='tight')
    print("Saved wind rose to wind_rose.png")
    plt.show()


if __name__ == '__main__':
    main()