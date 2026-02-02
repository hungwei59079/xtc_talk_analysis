import json
from pathlib import Path


class XTCConfig:
    """Wrapper for XTC analysis configuration.
    
    Provides a clean interface for accessing dataset configuration fields
    without requiring callers to know the internal JSON structure.
    
    Parameters
    ----------
    config_path : Path
        Path to the configuration JSON file.
    config_name : str
        Name of the dataset configuration to use.
    
    Examples
    --------
    >>> config = XTCConfig(Path("xtc_config.json"), "my_dataset")
    >>> config.xtc_dir
    '/path/to/xtc/data'
    >>> config.baseline_flags
    {'flag_datasets': ['is_baseline'], 'conditions': {'is_baseline': 63}}
    """
    
    def __init__(self, config_path: Path, config_name: str):
        self.config_path = Path(config_path)
        self.config_name = config_name
        
        with self.config_path.open() as f:
            full_config = json.load(f)
        
        if config_name not in full_config.get("datasets", {}):
            available = list(full_config.get("datasets", {}).keys())
            raise ValueError(
                f"Configuration '{config_name}' not found. "
                f"Available configurations: {available}"
            )
        
        self._config = full_config["datasets"][config_name]
    
    @property
    def xtc_dir(self) -> str:
        """Root directory for XTC data."""
        return self._config["xtc_dir"]
    
    @property
    def available_periods(self) -> dict:
        """Dictionary of available periods and their runs."""
        return self._config["available_periods"]
    
    @property
    def path_templates(self) -> dict:
        """Path templates for dsp_dir and hit_dir."""
        return self._config["path_templates"]
    
    @property
    def dsp_dir_template(self) -> str:
        """Template string for DSP directory paths."""
        return self._config["path_templates"]["dsp_dir"]
    
    @property
    def hit_dir_template(self) -> str:
        """Template string for hit directory paths."""
        return self._config["path_templates"]["hit_dir"]
    
    @property
    def baseline_flags(self) -> dict:
        """Baseline flag configuration with 'flag_datasets' and 'conditions'."""
        return self._config.get("baseline_flags", {})
    
    @property
    def baseline_flag_datasets(self) -> list:
        """List of flag dataset names for baseline selection."""
        return self.baseline_flags.get("flag_datasets", [])
    
    @property
    def xtalk_flags(self) -> dict:
        """Crosstalk flag configuration with 'flag_datasets' and 'conditions'."""
        return self._config.get("xtalk_flags", {})
    
    @property
    def xtalk_flag_trigger_datasets(self) -> list:
        """List of flag dataset names for crosstalk selection."""
        return self.xtalk_flags.get("trigger_datasets", [])
    
    @property
    def xtalk_flag_trigger_conditions(self) -> dict:
        """Dictionary of conditions for crosstalk trigger flag datasets."""
        return self.xtalk_flags.get("trigger_conditions", {})
    
    @property
    def xtalk_flag_response_datasets(self) -> list:
        """List of flag dataset names for crosstalk response selection."""
        return self.xtalk_flags.get("response_datasets", [])
    
    @property
    def xtalk_flag_response_conditions(self) -> dict:
        """Dictionary of conditions for crosstalk response flag datasets."""
        return self.xtalk_flags.get("response_conditions", {})
    
    @property
    def baseline_conditions(self) -> dict:
        """Dictionary of conditions for baseline flag datasets."""
        return self.baseline_flags.get("conditions", {})
    
    def get(self, key: str, default=None):
        """Get a configuration value by key.
        
        Use this for accessing optional or custom configuration fields
        that don't have dedicated properties.
        """
        return self._config.get(key, default)
    
    def __repr__(self) -> str:
        return f"XTCConfig(config_path={self.config_path!r}, config_name={self.config_name!r})"
