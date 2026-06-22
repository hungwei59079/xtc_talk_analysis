import matplotlib.pyplot as plt
import numpy as np
from lgdo import lh5

class EventSelector:
    def __init__(self, table_path, files, ene_dataset, conditions=None, energy_range=None, idx=None):
        flag_datasets = list(conditions.keys()) if conditions is not None else []
        all_fields = [ene_dataset] + flag_datasets
        table = lh5.read(table_path, files, field_mask=all_fields)
        self.energy_all = table[ene_dataset].nda

        if idx is not None:
            self.energy_all_indexed = self.energy_all[idx]
            selection_array = ~np.isnan(self.energy_all_indexed)
        else:
            self.energy_all_indexed = None
            selection_array = ~np.isnan(self.energy_all)
        
        for flag in flag_datasets:
            if idx is not None:
                flag_array = table[flag].nda[idx]
            else:
                flag_array = table[flag].nda
            condition = conditions.get(flag, True)
            selection_array &= (flag_array == condition)

        if energy_range is not None:
            emin, emax = energy_range
            if idx is not None:
                selection_array &= (self.energy_all_indexed >= emin) & (self.energy_all_indexed <= emax)
            else:
                selection_array &= (self.energy_all >= emin) & (self.energy_all <= emax)
        
        if idx is not None:
            self.selected_energies = self.energy_all_indexed[selection_array]
            self.selected_idxs = idx[selection_array]
        else:
            self.selected_energies = self.energy_all[selection_array]
            self.selected_idxs = np.arange(len(self.energy_all))
            self.selected_idxs = self.selected_idxs[selection_array]

    def draw(self, fig_path, y_scale='log', plot_range=None, bins_count=100):
        plt.figure(figsize=(10, 6))
        
        if plot_range is not None:
            xmin, xmax = plot_range
            # Filtering
            data_all = self.energy_all[(self.energy_all >= xmin) & (self.energy_all <= xmax)]
            data_sel = self.selected_energies[(self.selected_energies >= xmin) & (self.selected_energies <= xmax)]
        else:
            xmin, xmax = np.min(self.energy_all), np.max(self.energy_all)
            data_all = self.energy_all
            data_sel = self.selected_energies

        # 2. Calculate uniform bins strictly within this range
        bins = np.linspace(xmin, xmax, bins_count)
        
        # 3. Plot the histograms using the filtered data and common bins
        plt.hist(data_all, bins=bins, alpha=0.5, label=f"Pre-selection({len(self.energy_all)})")
        plt.hist(data_sel, bins=bins, alpha=0.5, label=f"Selected({len(self.selected_energies)})")
        
        # 4. Set the x-axis limits to match the requested range
        plt.xlim(xmin, xmax)
        
        plt.xlabel('Energy')
        plt.ylabel('Counts')
        plt.yscale(y_scale)
        plt.title('Energy Distribution Before and After Selection')
        plt.legend()
        plt.savefig(fig_path)
        plt.close()