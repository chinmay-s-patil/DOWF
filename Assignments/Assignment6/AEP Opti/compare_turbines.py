import os
import glob
import re
import matplotlib.pyplot as plt
import numpy as np

def main():
    directory = "/home/lavender/Studies/Design of Wind Farms/Assignments/Assignment6/optimizedLayout"
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found.")
        return

    # Nested dictionary to store AEP: data[n][turbine_type] = AEP
    data = {}
    turbine_types = set()

    # Read all layout text files
    for file_path in glob.glob(os.path.join(directory, "*_layout.txt")):
        filename = os.path.basename(file_path)
        
        # Regex to capture turbine type and n (handles names with underscores)
        match = re.search(r"^(.*)_(\d+)_layout\.txt$", filename)
        if match:
            t_name = match.group(1)
            n_val = int(match.group(2))
            
            turbine_types.add(t_name)
            
            with open(file_path, "r") as f:
                content = f.read()
                aep_match = re.search(r"AEP\s+:\s+([\d\.]+)", content)
                pen_match = re.search(r"Penalty\s+:\s+([\d\.]+)", content)
                
                if aep_match and pen_match:
                    aep = float(aep_match.group(1))
                    pen = float(pen_match.group(1))
                    
                    if n_val not in data:
                        data[n_val] = {}
                    data[n_val][t_name] = {"aep": aep, "pen": pen}

    if not data:
        print("No valid AEP/Penalty data found in the text files.")
        return

    # Sort n_values and turbine types for consistent plotting
    n_values = sorted(list(data.keys()))
    turbine_types = sorted(list(turbine_types))
    
    # Prepare the matrix for the grouped bar chart
    aep_matrix = {t: [] for t in turbine_types}
    pen_matrix = {t: [] for t in turbine_types}
    for n in n_values:
        for t in turbine_types:
            item = data[n].get(t, {"aep": 0.0, "pen": 0.0})
            aep_matrix[t].append(item["aep"])
            pen_matrix[t].append(item["pen"])

    # Generate the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(n_values))  # the label locations
    width = 0.8 / len(turbine_types)  # dynamic width based on number of types
    multiplier = 0
    
    # Plot each group
    for t_name in turbine_types:
        measurements = aep_matrix[t_name]
        penalties = pen_matrix[t_name]
        
        offset = width * multiplier
        # Center the groups around the x tick
        bar_x = x + offset - (width * (len(turbine_types) - 1) / 2)
        
        rects = ax.bar(bar_x, measurements, width, label=t_name)
        
        # Apply visual styling to bars that violate constraints
        for i, rect in enumerate(rects):
            if penalties[i] > 1e-3:
                rect.set_alpha(0.3)
                rect.set_hatch('//')
                rect.set_edgecolor('red')
                
        # Optional: Add text labels on top of the bars
        ax.bar_label(rects, padding=3, fmt='%.1f', fontsize=8)
        multiplier += 1

    ax.set_ylabel('Annual Energy Production (GWh/yr)')
    ax.set_xlabel('Number of Turbines (n)')
    ax.set_title('AEP Comparison by Turbine Count and Type')
    ax.set_xticks(x)
    ax.set_xticklabels(n_values)
    
    # Add constraint legend manually
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor='white', edgecolor='red', hatch='//', alpha=0.5))
    labels.append('Violates Constraints (Penalty > 1e-3)')
    ax.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Add some headroom for the labels
    max_aep = max([max(v) for v in aep_matrix.values() if v])
    ax.set_ylim(0, max_aep * 1.15)

    plt.tight_layout()
    out_file = "/home/lavender/Studies/Design of Wind Farms/Assignments/Assignment6/AEP_comparison.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {out_file}")

if __name__ == "__main__":
    main()
