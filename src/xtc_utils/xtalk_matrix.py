import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

reason_dict = {"no_stats" : "no_stats",
               "low_stats" : "low_stats",
               "Optimal parameters not found: Number of calls to function has reached maxfev = 800.": "delta" ,
               "ok but with insufficient points" : "sharp"}

class XTCMatrix:
    def __init__(self, n_detectors, label, **kwargs):
        # Data Related
        self.n_detectors = n_detectors
        self.matrix = kwargs.get("matrix", np.full((n_detectors, n_detectors), np.nan))
        self.sigma_matrix = kwargs.get("sigma_matrix", np.full((n_detectors, n_detectors), np.nan))
        self.fail_dict = kwargs.get("fail_dict", {"no_stats": [], "low_stats": [], "delta": [], "sharp": []})
        self.scenarios = kwargs.get("scenarios", set())

        # I/O Related
        self.load_path = Path(kwargs.get("load_path")) if kwargs.get("load_path", None) else None
        self.save_path = Path(kwargs.get("save_path")) if kwargs.get("save_path", None) else None
        self.imagename = kwargs.get("filename", f"{label}_xtalk_matrix.png")
        self.csvname = kwargs.get("csvname", f"{label}_xtalk_matrix.csv")
        self.sigma_csvname = kwargs.get("sigma_csvname", f"{label}_xtalk_matrix_sigma.csv")

        # Plot Related
        self.label = label
        if label in ["neg", "neg_restrained"]:
            self.vmin = kwargs.get("vmin", -0.3)
            self.vmax = kwargs.get("vmax", 0.1)
        elif label in ["pos", "pos_restrained"]:
            self.vmin = kwargs.get("vmin", -0.07)
            self.vmax = kwargs.get("vmax", 0.3)
        else:
            self.vmin = kwargs.get("vmin", None)
            self.vmax = kwargs.get("vmax", None)
        self.title = kwargs.get("title", f"{label} Crosstalk Matrix Heatmap")
        self.cbar_label = kwargs.get("cbar_label", "Crosstalk Value (%)")
        

    def load(self, chn_id, skipped_channels, path=None):
        self.load_path = Path(path) if path is not None else self.load_path
        if self.load_path is None:
            raise ValueError("Load path is not specified.")
        
        if len(chn_id) != self.n_detectors:
            raise ValueError("Length of chn_id does not match n_detectors.")
        
        for j1 in range(self.n_detectors):
            raw_id_1 = chn_id[j1]
            is_skipped_1 = raw_id_1 in skipped_channels
            for j2 in range(self.n_detectors):
                raw_id_2 = chn_id[j2]
                is_skipped_2 = raw_id_2 in skipped_channels
                npz_path = self.load_path / f"fit_{self.label}_{j1}_{j2}.npz"
                try:
                    with np.load(npz_path) as data:
                        reason = str(data["reason"])
                        mu = float(data["mu"])
                        sigma = float(data["sigma"])
                except Exception as e:
                    print(f"Exception {e} occurs. Skipping.")
                    continue
                if reason not in self.scenarios:
                    self.scenarios.add(reason)
                self.matrix[j1,j2] = mu
                self.sigma_matrix[j1,j2] = sigma
                if reason in reason_dict:
                    abb_reason = reason_dict[reason]
                    if not is_skipped_1 and not is_skipped_2 and raw_id_1 != raw_id_2:
                        self.fail_dict[abb_reason].append(f"({j1},{j2})")
    
    def plot(self, path=None, cmap=plt.cm.jet_r):
        self.save_path = Path(path) if path is not None else self.save_path
        plt.figure(figsize=(8, 6))
        im = plt.imshow(self.matrix, origin="lower", vmin=self.vmin, vmax=self.vmax, cmap=cmap)
        plt.colorbar(im, label=self.cbar_label)

        plt.xlabel('Response Channel Index')
        plt.ylabel('Trigger Channel Index')
        plt.title(self.title)
        plt.tight_layout()
        plt.savefig(self.save_path / self.imagename)
        plt.close()

    def save_csv(self, path=None):
        self.save_path = Path(path) if path is not None else self.save_path
        np.savetxt(self.save_path / self.csvname, self.matrix, delimiter=",", fmt="%.6f")
        np.savetxt(self.save_path / self.sigma_csvname, self.sigma_matrix, delimiter=",", fmt="%.6f")

    def diagnose(self):
        print(f"Scenarios encountered for {self.label} matrix: {self.scenarios}")
        for reason, channels in self.fail_dict.items():
            print(f"Reason: {reason}, Failed Channel Pairs: {channels}")

    

