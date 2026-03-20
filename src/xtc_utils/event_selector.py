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

    def draw(self, fig_path, y_scale='log'):
        plt.figure(figsize=(10, 6))
        plt.hist(self.energy_all, bins=100, alpha=0.5, label=f"Pre-selection({len(self.energy_all)})")
        plt.hist(self.selected_energies, bins=100, alpha=0.5, label=f"Selected({len(self.selected_energies)})")
        plt.xlabel('Energy')
        plt.ylabel('Counts')
        plt.yscale(y_scale)
        plt.title('Energy Distribution Before and After Selection')
        plt.legend()
        plt.savefig(fig_path)
        plt.close()