import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from lgdo import lh5, types

reason_dict = {"no_stats" : "no_stats",
               "low_stats" : "low_stats",
               "Optimal parameters not found: Number of calls to function has reached maxfev = 800.": "delta" ,
               "ok but with insufficient points" : "sharp"}

# Maps the label prefix onto the field name used in the production xtc files.
# "neg" and "neg_restrained" both land on "xtalk_matrix_negative".
lh5_field_dict = {"neg": "xtalk_matrix_negative",
                  "pos": "xtalk_matrix_positive"}

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
    
    def plot(self, path=None, filename=None, cmap=plt.cm.jet_r):
        save_path = Path(path) if path is not None else Path.cwd()
        filename = filename if filename is not None else f"{self.label}_xtalk_matrix.png"

        plt.figure(figsize=(8, 6))
        im = plt.imshow(self.matrix, origin="lower", vmin=self.vmin, vmax=self.vmax, cmap=cmap)
        plt.colorbar(im, label=self.cbar_label)

        plt.xlabel('Response Channel Index')
        plt.ylabel('Trigger Channel Index')
        plt.title(self.title)
        plt.tight_layout()
        plt.savefig(save_path / filename)
        plt.close()

    def save_csv(self, path=None, filename=None, sigma_filename=None):
        save_path = Path(path) if path is not None else Path.cwd()
        filename = filename if filename is not None else f"{self.label}_xtalk_matrix.csv"
        sigma_filename = (sigma_filename if sigma_filename is not None
                          else f"{self.label}_xtalk_matrix_sigma.csv")

        np.savetxt(save_path / filename, self.matrix, delimiter=",", fmt="%.6f")
        np.savetxt(save_path / sigma_filename, self.sigma_matrix, delimiter=",", fmt="%.6f")

    @property
    def lh5_field(self):
        """Field name this label is stored under in an xtc lh5 file."""
        for prefix, field in lh5_field_dict.items():
            if self.label.startswith(prefix):
                return field
        raise ValueError(
            f"Label {self.label!r} does not start with any of "
            f"{list(lh5_field_dict)}, so there is no lh5 field name for it. "
            f"Pass field=... to save_lh5() to name it explicitly."
        )

    def save_lh5(self, chn_id, path=None, filename=None, group="xtc",
                 field=None, save_sigma=True, in_percent=True):
        """Write the matrix into an lh5 file in the production xtc layout.

        The file holds a single table (``group``) whose columns are
        ``rawid_index`` (the rawid of the detector at each matrix row/column
        index) plus one 2D matrix per label. Calling this on a file that
        already has the table appends the new column(s) instead of replacing
        it, so several XTCMatrix objects can write into one file. Re-writing a
        column that is already present overwrites it.

        Parameters
        ----------
        chn_id : sequence of int
            rawids in matrix-index order, i.e. ``chn_id[j]`` is the detector
            at row/column ``j``. This is the ``chn_id`` returned by
            :func:`files_and_chnid`, and it is stored as ``rawid_index``.
        path : str or Path, optional
            Directory to write into. Defaults to the current directory.
        filename : str, optional
            File name. Defaults to ``"par_evt_xtc.lh5"``.
        group : str, optional
            Table name inside the file. Default ``"xtc"``.
        field : str, optional
            Column name for the matrix. Defaults to the name implied by the
            label (see :attr:`lh5_field`).
        save_sigma : bool, optional
            Also write the fit uncertainties as ``{field}_sigma``. This is an
            extra column the production files do not have. Default True.
        in_percent : bool, optional
            Whether ``self.matrix`` is in percent. Production xtc files store
            fractions, so when True (the default) the values are divided by
            100 on the way out.
        """
        save_path = Path(path) if path is not None else Path.cwd()
        filename = filename if filename is not None else "par_evt_xtc.lh5"
        field = field if field is not None else self.lh5_field

        chn_id = np.asarray(chn_id, dtype=np.int64)
        if len(chn_id) != self.n_detectors:
            raise ValueError(
                f"Length of chn_id ({len(chn_id)}) does not match n_detectors "
                f"({self.n_detectors})."
            )

        scale = 0.01 if in_percent else 1.0
        columns = {field: types.Array(self.matrix * scale)}
        if save_sigma:
            columns[f"{field}_sigma"] = types.Array(self.sigma_matrix * scale)

        lh5_path = save_path / filename
        table = None
        if lh5_path.exists() and group in lh5.ls(str(lh5_path)):
            table = lh5.read(group, str(lh5_path))
            existing = table["rawid_index"].nda
            if not np.array_equal(existing, chn_id):
                raise ValueError(
                    f"{lh5_path} already holds a '{group}' table whose "
                    f"rawid_index differs from the chn_id passed in. Refusing "
                    f"to append a matrix whose rows mean something different "
                    f"from the ones already in the file."
                )
            for name, column in columns.items():
                if name in table.keys():
                    table.remove_column(name)
                table.add_column(name, column)

        if table is None:
            columns = {"rawid_index": types.Array(chn_id), **columns}
            table = types.Table(col_dict=columns)
            lh5.write(table, group, str(lh5_path), wo_mode="write_safe")
        else:
            lh5.write(table, group, str(lh5_path), wo_mode="overwrite")

        print(f"Wrote {self.label} matrix to {lh5_path}:{group}/{field}")

    def diagnose(self):
        print(f"Scenarios encountered for {self.label} matrix: {self.scenarios}")
        for reason, channels in self.fail_dict.items():
            print(f"Reason: {reason}, Failed Channel Pairs: {channels}")

    

