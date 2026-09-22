"""
Observation class for Bfrescox

"""

import tempfile
from pathlib import Path

import bfrescox
import numpy as np
from pandas import DataFrame

from .observation import Observation


class BfrescoxObservation(Observation):
    """
    Observation that represents a single output from a Bfrescox calculation,
    such as an elastic differential cross section
    (dXS/dΩ, dXS/dRuth, or analysing power Ay) at energy and set of angles.

    TODO: allow for multiple observations from a single bfrescox run
    (e.g. Ay and dXS/dΩ at the same energy and angles)
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        template_path: Path | str,
        runtime_path: Path | str | None = None,
        y_stat_err=None,
        y_sys_err_normalization=None,
        y_sys_err_normalization_mask=None,
        y_sys_err_offset=None,
        y_sys_err_offset_mask=None,
    ):
        self.template_path = Path(template_path)
        self.runtime_path = (
            Path(runtime_path) if runtime_path is not None else Path("./")
        )

        super().__init__(
            x,
            y,
            y_stat_err=y_stat_err,
            y_sys_err_normalization=y_sys_err_normalization,
            y_sys_err_normalization_mask=y_sys_err_normalization_mask,
            y_sys_err_offset=y_sys_err_offset,
            y_sys_err_offset_mask=y_sys_err_offset_mask,
        )

    def run(self, params_dict: dict) -> dict[str, DataFrame]:
        """
        Run the Bfrescox calculation with the given parameters
        and return the predicted observable as a DataFrame.
        """
        with tempfile.TemporaryDirectory(dir=self.runtime_path) as tmp_dir_str:
            run_dir= Path(tmp_dir_str)
            
            cfg = bfrescox.Configuration.from_template(
                self.template_path,
                run_dir / "frescox.in",
                params_dict,
                overwrite=True,
            )
            bfrescox.run_simulation(
                cfg,
                run_dir / "frescox.out",
                cwd=run_dir,
                overwrite=True,
            )
            results = bfrescox.parse_fort16(run_dir / "fort.16")

        return results


class BfrescoxCombinedObservation(Observation):
    """
    Observation that represents a single output from a Bfrescox calculation,
    such as an elastic differential cross section
    (dXS/dΩ, dXS/dRuth, or analysing power Ay) at energy and set of angles.

    TODO: allow for multiple observations from a single bfrescox run
    (e.g. Ay and dXS/dΩ at the same energy and angles)
    """

    def __init__(
        self,
        x1: np.ndarray,
        y1: np.ndarray,
        x2: np.ndarray,
        y2: np.ndarray,
        template_path: Path | str,
        runtime_path: Path | str | None = None,
        y1_stat_err=None,
        y2_stat_err=None,
        y1_sys_err_normalization=None,

    ):
        self.template_path = Path(template_path)
        self.runtime_path = (
            Path(runtime_path) if runtime_path is not None else Path("./")
        )
        self.x1 = x1
        self.x2 = x2
        
        x_combined = np.concatenate((x1, x2))
        y_combined = np.concatenate((y1, y2))
        y_stat_err_combined = np.concatenate((y1_stat_err, y2_stat_err))

        mask_sigma = np.concatenate((
            np.ones(len(x1), dtype=bool),
            np.zeros(len(x2), dtype=bool),
        ))

        if y1_sys_err_normalization is not None:
            super().__init__(
            x_combined,
            y_combined,
            y_stat_err=y_stat_err_combined,
            y_sys_err_normalization=[y1_sys_err_normalization],
            y_sys_err_normalization_mask=[mask_sigma],
            )

        else:
            super().__init__(
            x_combined,
            y_combined,
            y_stat_err=y_stat_err_combined,
            )
                    

    def run(self, params_dict: dict) -> dict[str, DataFrame]:
        """
        Run the Bfrescox calculation with the given parameters
        and return the predicted observable as a DataFrame.
        """
        with tempfile.TemporaryDirectory(dir=self.runtime_path) as tmp_dir_str:
            run_dir= Path(tmp_dir_str)
            
            cfg = bfrescox.Configuration.from_template(
                self.template_path,
                run_dir / "frescox.in",
                params_dict,
                overwrite=True,
            )
            bfrescox.run_simulation(
                cfg,
                run_dir / "frescox.out",
                cwd=run_dir,
                overwrite=True,
            )
            results = bfrescox.parse_fort16(run_dir / "fort.16")

        return results    





class BfrescoxCombinedObservation(Observation):
    """
    Observation that represents a single output from a Bfrescox calculation,
    such as an elastic differential cross section
    (dXS/dΩ, dXS/dRuth, or analysing power Ay) at energy and set of angles.

    TODO: allow for multiple observations from a single bfrescox run
    (e.g. Ay and dXS/dΩ at the same energy and angles)
    """

    def __init__(
        self,
        x1: np.ndarray,
        y1: np.ndarray,
        x2: np.ndarray,
        y2: np.ndarray,
        template_path: Path | str,
        runtime_path: Path | str | None = None,
        y1_stat_err=None,
        y2_stat_err=None,
        y1_sys_err_normalization=None,

    ):
        self.template_path = Path(template_path)
        self.runtime_path = (
            Path(runtime_path) if runtime_path is not None else Path("./")
        )
        self.x1 = x1
        self.x2 = x2
        
        x_combined = np.concatenate((x1, x2))
        y_combined = np.concatenate((y1, y2))
        y_stat_err_combined = np.concatenate((y1_stat_err, y2_stat_err))

        mask_sigma = np.concatenate((
            np.ones(len(x1), dtype=bool),
            np.zeros(len(x2), dtype=bool),
        ))

        if y1_sys_err_normalization is not None:
            super().__init__(
            x_combined,
            y_combined,
            y_stat_err=y_stat_err_combined,
            y_sys_err_normalization=[y1_sys_err_normalization],
            y_sys_err_normalization_mask=[mask_sigma],
            )

        else:
            super().__init__(
            x_combined,
            y_combined,
            y_stat_err=y_stat_err_combined,
            )
                    

    def run(self, params_dict: dict) -> dict[str, DataFrame]:
        """
        Run the Bfrescox calculation with the given parameters
        and return the predicted observable as a DataFrame.
        """
        with tempfile.TemporaryDirectory(dir=self.runtime_path) as tmp_dir_str:
            run_dir= Path(tmp_dir_str)
            
            cfg = bfrescox.Configuration.from_template(
                self.template_path,
                run_dir / "frescox.in",
                params_dict,
                overwrite=True,
            )
            bfrescox.run_simulation(
                cfg,
                run_dir / "frescox.out",
                cwd=run_dir,
                overwrite=True,
            )
            results = bfrescox.parse_fort16(run_dir / "fort.16")

        return results        



class BfrescoxTransferObservation(Observation):
    """
    Observation that represents a single output from a Bfrescox calculation,
    such as an elastic differential cross section
    (dXS/dΩ, dXS/dRuth, or analysing power Ay) at energy and set of angles.

    TODO: allow for multiple observations from a single bfrescox run
    (e.g. Ay and dXS/dΩ at the same energy and angles)
    """

    def __init__(
        self,
        x1: np.ndarray,
        y1: np.ndarray,
        x2: np.ndarray,
        y2: np.ndarray,
        x3: np.ndarray,
        y3: np.ndarray,
        template_path: Path | str,
        runtime_path: Path | str | None = None,
        y1_stat_err=None,
        y2_stat_err=None,
        y3_stat_err=None,

    ):
        self.template_path = Path(template_path)
        self.runtime_path = (
            Path(runtime_path) if runtime_path is not None else Path("./")
        )
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        
        x_combined = np.concatenate((x1, x2, x3))
        y_combined = np.concatenate((y1, y2, y3))
        y_stat_err_combined = np.concatenate((y1_stat_err, y2_stat_err, y3_stat_err))

        super().__init__(
            x_combined,
            y_combined,
            y_stat_err=y_stat_err_combined,
            )    
                    

    def run(self, params_dict: dict) -> dict[str, DataFrame]:
    
       """
        Run the Bfrescox calculation with the given parameters
        and return the predicted observable as a DataFrame.
        """
       
       with tempfile.TemporaryDirectory(dir=self.runtime_path) as tmp_dir_str:
            run_dir= Path(tmp_dir_str)
            
            cfg = bfrescox.Configuration.from_template(
                self.template_path,
                run_dir / "frescox.in",
                params_dict,
                overwrite=True,
            )
            bfrescox.run_simulation(
                cfg,
                run_dir / "frescox.out",
                cwd=run_dir,
                overwrite=True,
            )
            results = bfrescox.parse_fort16(run_dir / "fort.16")

       return results


    def run_single_ANC(self, params_dict: dict) -> dict[str, DataFrame]:
        
        """
        Run the Bfrescox calculation with the given parameters and return the single ANC
        """

        with tempfile.TemporaryDirectory(dir=self.runtime_path) as tmp_dir_str:
            run_dir= Path(tmp_dir_str)
            
            cfg = bfrescox.Configuration.from_template(
                self.template_path,
                run_dir / "frescox.in",
                params_dict,
                overwrite=True,
            )
            bfrescox.run_simulation(
                cfg,
                run_dir / "frescox.out",
                cwd=run_dir,
                overwrite=True,
            )
            
            single_ANC = np.loadtxt(
                run_dir / "fort.46",
                usecols=1,
                max_rows=1
            )

        return single_ANC


        




class BfrescoxTransferObservation2(Observation):
    """
    Observation that represents a single output from a Bfrescox calculation,
    such as an elastic differential cross section
    (dXS/dΩ, dXS/dRuth, or analysing power Ay) at energy and set of angles.

    TODO: allow for multiple observations from a single bfrescox run
    (e.g. Ay and dXS/dΩ at the same energy and angles)
    """

    def __init__(
        self,
        x1: np.ndarray,
        y1: np.ndarray,
        x2: np.ndarray,
        y2: np.ndarray,
        template_path: Path | str,
        runtime_path: Path | str | None = None,
        y1_stat_err=None,
        y2_stat_err=None,

    ):
        self.template_path = Path(template_path)
        self.runtime_path = (
            Path(runtime_path) if runtime_path is not None else Path("./")
        )
        self.x1 = x1
        self.x2 = x2
        
        x_combined = np.concatenate((x1, x2))
        y_combined = np.concatenate((y1, y2))
        y_stat_err_combined = np.concatenate((y1_stat_err, y2_stat_err))

        super().__init__(
            x_combined,
            y_combined,
            y_stat_err=y_stat_err_combined,
            )    
                    

    def run(self, params_dict: dict) -> dict[str, DataFrame]:
    
       """
        Run the Bfrescox calculation with the given parameters
        and return the predicted observable as a DataFrame.
        """
       
       with tempfile.TemporaryDirectory(dir=self.runtime_path) as tmp_dir_str:
            run_dir= Path(tmp_dir_str)
            
            cfg = bfrescox.Configuration.from_template(
                self.template_path,
                run_dir / "frescox.in",
                params_dict,
                overwrite=True,
            )
            bfrescox.run_simulation(
                cfg,
                run_dir / "frescox.out",
                cwd=run_dir,
                overwrite=True,
            )
            results = bfrescox.parse_fort16(run_dir / "fort.16")

       return results


    def run_single_ANC(self, params_dict: dict) -> dict[str, DataFrame]:
        
        """
        Run the Bfrescox calculation with the given parameters and return the single ANC
        """

        with tempfile.TemporaryDirectory(dir=self.runtime_path) as tmp_dir_str:
            run_dir= Path(tmp_dir_str)
            
            cfg = bfrescox.Configuration.from_template(
                self.template_path,
                run_dir / "frescox.in",
                params_dict,
                overwrite=True,
            )
            bfrescox.run_simulation(
                cfg,
                run_dir / "frescox.out",
                cwd=run_dir,
                overwrite=True,
            )
            
            single_ANC = np.loadtxt(
                run_dir / "fort.46",
                usecols=1,
                max_rows=1
            )

        return single_ANC      
    
            
