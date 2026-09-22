"""
Physical model for elastic differential cross sections.

:class:`ElasticDifferentialXSModel` wraps a ``jitr`` optical-model solver to
predict elastic differential cross sections (dXS/dΩ, dXS/dRuth, or analysing
power Ay) given a parametric central and spin-orbit interaction.
"""

import numpy as np

from .bfrescox_observation import BfrescoxObservation, BfrescoxCombinedObservation, BfrescoxTransferObservation, BfrescoxTransferObservation2
from .observation import Observation
from .physical_model import PhysicalModel


class BfrescoxModel(PhysicalModel):
    """
    A model that runs Bfrescox to predict reaction observables
    """

    def __init__(
        self,
        params: list = [],
        model_name: str | None = None,
        channel_name: str | None = None,
    ):
        """
        Parameters
        ----------
        params : list of Parameter, optional
            Parameters of the model.  Defaults to ``[]``.
        model_name : str, optional
            Human-readable model name.  Defaults to ``"ElasticDifferentialXSModel"``.
        """
        self.model_name = model_name or "BfrescoxModel"
        self.channel_name = channel_name or "channel_1"

        super().__init__(params)

    def evaluate(
        self,
        observation: Observation,
        *params: tuple,
    ) -> np.ndarray:
        """ """
        if not isinstance(observation, BfrescoxObservation):
            raise ValueError(
                f"Observation must be a BfrescoxObservation, but got {type(observation)}"
            )

        # convert params tuple to dict
        params_dict = {param.name: val for param, val in zip(self.params, params)}

        # run the Bfrescox calculation and return the predicted observable
        df = observation.run(params_dict)[self.channel_name]

        # extract the predicted observable from the DataFrame and return it
        # as a numpy array

        theta_fresco=df["Theta_deg"].to_numpy()
        sigma_fresco=df["sigma_mb_sr"].to_numpy()

        sigma_interp = np.interp(
            observation.x,
            theta_fresco,
            sigma_fresco,
        )
        
        return sigma_interp



    def visualizable_model_prediction(
        self,
        observation: Observation,
        *params: tuple,
    ) -> np.ndarray:
        """ """
        if not isinstance(observation, BfrescoxObservation):
            raise ValueError(
                f"Observation must be a BfrescoxObservation, but got {type(observation)}"
            )

        # convert params tuple to dict
        params_dict = {param.name: val for param, val in zip(self.params, params)}

        # run the Bfrescox calculation and return the predicted observable
        df = observation.run(params_dict)[self.channel_name]

        # extract the predicted observable from the DataFrame and return it
        # as a numpy array

        theta_vis=df["Theta_deg"].to_numpy()
        sigma_vis=df["sigma_mb_sr"].to_numpy()
        
        return theta_vis, sigma_vis




class BfrescoxModelCombined(PhysicalModel):
    """
    A model that combine cross sections and analyzing power and runs Bfrescox to predict reaction observables
    """

    def __init__(
        self,
        params: list = [],
        model_name: str | None = None,
        channel_name: str | None = None,
    ):
        """
        Parameters
        ----------
        params : list of Parameter, optional
            Parameters of the model.  Defaults to ``[]``.
        model_name : str, optional
            Human-readable model name.  Defaults to ``"ElasticDifferentialXSModel"``.
        """
        self.model_name = model_name or "BfrescoxCombinedModel"
        self.channel_name = channel_name or "channel_1"

        super().__init__(params)

    def evaluate(
        self,
        observation: Observation,
        *params: tuple,
    ) -> np.ndarray:
        """ """
        if not isinstance(observation, BfrescoxCombinedObservation):
            raise ValueError(
                f"Observation must be a BfrescoxCombinedObservation, but got {type(observation)}"
            )

        # convert params tuple to dict
        params_dict = {param.name: val for param, val in zip(self.params, params)}

        # run the Bfrescox calculation and return the predicted observable
        df = observation.run(params_dict)[self.channel_name]

        # extract the predicted observable from the DataFrame and return it
        # as a numpy array

        theta_fresco=df["Theta_deg"].to_numpy()
        sigma_fresco=df["sigma_mb_sr"].to_numpy()
        iT11_fresco=df["iT11"].to_numpy()


        sigma_interp = np.interp(
            observation.x1,
            theta_fresco,
            sigma_fresco,
        )

        iT11_interp = np.interp(
            observation.x2,
            theta_fresco,
            iT11_fresco,
        )

        interp_combined = np.concatenate((sigma_interp, iT11_interp))
        
        return interp_combined



    def visualizable_model_prediction(
        self,
        observation: Observation,
        *params: tuple,
    ) -> np.ndarray:
        """ """
        if not isinstance(observation, BfrescoxCombinedObservation):
            raise ValueError(
                f"Observation must be a BfrescoxCombinedObservation, but got {type(observation)}"
            )

        # convert params tuple to dict
        params_dict = {param.name: val for param, val in zip(self.params, params)}

        # run the Bfrescox calculation and return the predicted observable
        df = observation.run(params_dict)[self.channel_name]

        # extract the predicted observable from the DataFrame and return it
        # as a numpy array

        theta_vis=df["Theta_deg"].to_numpy()
        sigma_vis=df["sigma_mb_sr"].to_numpy()
        iT11_vis=df["iT11"].to_numpy()
        
        return theta_vis, sigma_vis, iT11_vis   


    

class BfrescoxModelNormalisation(PhysicalModel):
    """
    A model that runs Bfrescox to predict reaction observables
    """

    def __init__(
        self,
        params: list = [],
        model_name: str | None = None,
        channel_name: str | None = None,
    ):
        """
        Parameters
        ----------
        params : list of Parameter, optional
            Parameters of the model.  Defaults to ``[]``.
        model_name : str, optional
            Human-readable model name.  Defaults to ``"ElasticDifferentialXSModel"``.
        """
        self.model_name = model_name or "BfrescoxModel"
        self.channel_name = channel_name or "channel_1"

        super().__init__(params)

    def evaluate(
        self,
        observation: Observation,
        *params: tuple,
    ) -> np.ndarray:
        """ """
        if not isinstance(observation, BfrescoxObservation):
            raise ValueError(
                f"Observation must be a BfrescoxObservation, but got {type(observation)}"
            )

        # convert params tuple to dict
        params_dict = {param.name: val for param, val in zip(self.params, params)}

        N = params_dict.pop("N")

        # run the Bfrescox calculation and return the predicted observable
        df = observation.run(params_dict)[self.channel_name]

        # extract the predicted observable from the DataFrame and return it
        # as a numpy array

        theta_fresco=df["Theta_deg"].to_numpy()
        sigma_fresco=df["sigma_mb_sr"].to_numpy()

        sigma_interp = np.interp(
            observation.x,
            theta_fresco,
            sigma_fresco,
        )

        sigma_interp = N * sigma_interp
        
        return sigma_interp



    def visualizable_model_prediction(
        self,
        observation: Observation,
        *params: tuple,
    ) -> np.ndarray:
        """ """
        if not isinstance(observation, BfrescoxObservation):
            raise ValueError(
                f"Observation must be a BfrescoxObservation, but got {type(observation)}"
            )

        # convert params tuple to dict
        params_dict = {param.name: val for param, val in zip(self.params, params)}

        N = params_dict.pop("N")

        # run the Bfrescox calculation and return the predicted observable
        df = observation.run(params_dict)[self.channel_name]

        # extract the predicted observable from the DataFrame and return it
        # as a numpy array

        theta_vis=df["Theta_deg"].to_numpy()
        sigma_vis=df["sigma_mb_sr"].to_numpy()

        sigma_vis = N * sigma_vis
        
        return theta_vis, sigma_vis



class BfrescoxModelTransfer(PhysicalModel):
    """
    A model that combine cross sections and analyzing power and runs Bfrescox to predict reaction observables
    """

    def __init__(
        self,
        params: list = [],
        model_name: str | None = None,
        channel_name: str | None = None,
    ):
        """
        Parameters
        ----------
        params : list of Parameter, optional
            Parameters of the model.  Defaults to ``[]``.
        model_name : str, optional
            Human-readable model name.  Defaults to ``"ElasticDifferentialXSModel"``.
        """
        self.model_name = model_name or "BfrescoxModelTransfer"

        super().__init__(params)

    def evaluate(
        self,
        observation: Observation,
        *params: tuple,
    ) -> np.ndarray:
        """ """
        if not isinstance(observation, BfrescoxTransferObservation):
            raise ValueError(
                f"Observation must be a BfrescoxTransferObservation, but got {type(observation)}"
            )

        # convert params tuple to dict
        params_dict = {param.name: val for param, val in zip(self.params, params)}

        N = params_dict.pop("N")

        results = observation.run(params_dict)

        # run the Bfrescox calculation and return the predicted observable
        df_elastic = results["channel_1"]
        df_transfer = results["channel_2"]

        # extract the predicted observable from the DataFrame and return it
        # as a numpy array

        theta_elastic = df_elastic["Theta_deg"].to_numpy()
        sigma_elastic = df_elastic["sigma_mb_sr"].to_numpy()
        iT11_fresco = df_elastic["iT11"].to_numpy()


        sigma_elastic_interp = np.interp(
            observation.x1,
            theta_elastic,
            sigma_elastic,
        )

        iT11_interp = np.interp(
            observation.x2,
            theta_elastic,
            iT11_fresco,
        )

        theta_transfer = df_transfer["Theta_deg"].to_numpy()
        sigma_transfer = df_transfer["sigma_mb_sr"].to_numpy()


        sigma_transfer_interp = np.interp(
            observation.x3,
            theta_transfer,
            sigma_transfer, 
        )

        sigma_transfer_interp = N * sigma_transfer_interp
        
        interp_combined = np.concatenate((sigma_elastic_interp, iT11_interp, sigma_transfer_interp))
        
        return interp_combined



    def visualizable_model_prediction(
        self,
        observation: Observation,
        *params: tuple,
    ) -> np.ndarray:
        """ """
        if not isinstance(observation, BfrescoxTransferObservation):
            raise ValueError(
                f"Observation must be a BfrescoxTransferObservation, but got {type(observation)}"
            )

        # convert params tuple to dict
        params_dict = {param.name: val for param, val in zip(self.params, params)}

        N = params_dict.pop("N")

        results = observation.run(params_dict)

        # run the Bfrescox calculation and return the predicted observable
        df_elastic = results["channel_1"]
        df_transfer = results["channel_2"]

        # extract the predicted observable from the DataFrame and return it
        # as a numpy array

        theta_elastic_vis = df_elastic["Theta_deg"].to_numpy()
        sigma_elastic_vis = df_elastic["sigma_mb_sr"].to_numpy()
        iT11_vis = df_elastic["iT11"].to_numpy()

        theta_transfer_vis = df_transfer["Theta_deg"].to_numpy()
        sigma_transfer_vis = df_transfer["sigma_mb_sr"].to_numpy()

        sigma_transfer_vis = N * sigma_transfer_vis
        
        return theta_elastic_vis, sigma_elastic_vis, iT11_vis, theta_transfer_vis, sigma_transfer_vis



    def single_ANC(
        self,
        observation: Observation,
        *params: tuple,
    ) -> np.ndarray:
        """ """
        if not isinstance(observation, BfrescoxTransferObservation):
            raise ValueError(
                f"Observation must be a BfrescoxTransferObservation, but got {type(observation)}"
            )

        # convert params tuple to dict
        params_dict = {param.name: val for param, val in zip(self.params, params)}

        N = params_dict.pop("N")

        ANC = observation.run_single_ANC(params_dict)

        return ANC
        


class BfrescoxModelTransfer2(PhysicalModel):
    """
    A model that combine cross sections and analyzing power and runs Bfrescox to predict reaction observables
    """

    def __init__(
        self,
        params: list = [],
        model_name: str | None = None,
        channel_name: str | None = None,
    ):
        """
        Parameters
        ----------
        params : list of Parameter, optional
            Parameters of the model.  Defaults to ``[]``.
        model_name : str, optional
            Human-readable model name.  Defaults to ``"ElasticDifferentialXSModel"``.
        """
        self.model_name = model_name or "BfrescoxModelTransfer"

        super().__init__(params)

    def evaluate(
        self,
        observation: Observation,
        *params: tuple,
    ) -> np.ndarray:
        """ """
        if not isinstance(observation, BfrescoxTransferObservation2):
            raise ValueError(
                f"Observation must be a BfrescoxTransferObservation2, but got {type(observation)}"
            )

        # convert params tuple to dict
        params_dict = {param.name: val for param, val in zip(self.params, params)}

        N = params_dict.pop("N")

        results = observation.run(params_dict)

        # run the Bfrescox calculation and return the predicted observable
        df_elastic = results["channel_1"]
        df_transfer = results["channel_2"]

        # extract the predicted observable from the DataFrame and return it
        # as a numpy array

        theta_elastic = df_elastic["Theta_deg"].to_numpy()
        sigma_elastic = df_elastic["sigma_mb_sr"].to_numpy()


        sigma_elastic_interp = np.interp(
            observation.x1,
            theta_elastic,
            sigma_elastic,
        )


        theta_transfer = df_transfer["Theta_deg"].to_numpy()
        sigma_transfer = df_transfer["sigma_mb_sr"].to_numpy()


        sigma_transfer_interp = np.interp(
            observation.x2,
            theta_transfer,
            sigma_transfer, 
        )

        sigma_transfer_interp = N * sigma_transfer_interp
        
        interp_combined = np.concatenate((sigma_elastic_interp, sigma_transfer_interp))
        
        return interp_combined



    def visualizable_model_prediction(
        self,
        observation: Observation,
        *params: tuple,
    ) -> np.ndarray:
        """ """
        if not isinstance(observation, BfrescoxTransferObservation2):
            raise ValueError(
                f"Observation must be a BfrescoxTransferObservation2, but got {type(observation)}"
            )

        # convert params tuple to dict
        params_dict = {param.name: val for param, val in zip(self.params, params)}

        N = params_dict.pop("N")

        results = observation.run(params_dict)

        # run the Bfrescox calculation and return the predicted observable
        df_elastic = results["channel_1"]
        df_transfer = results["channel_2"]

        # extract the predicted observable from the DataFrame and return it
        # as a numpy array

        theta_elastic_vis = df_elastic["Theta_deg"].to_numpy()
        sigma_elastic_vis = df_elastic["sigma_mb_sr"].to_numpy()

        theta_transfer_vis = df_transfer["Theta_deg"].to_numpy()
        sigma_transfer_vis = df_transfer["sigma_mb_sr"].to_numpy()

        sigma_transfer_vis = N * sigma_transfer_vis
        
        return theta_elastic_vis, sigma_elastic_vis, theta_transfer_vis, sigma_transfer_vis    



    def single_ANC(
        self,
        observation: Observation,
        *params: tuple,
    ) -> np.ndarray:
        """ """
        if not isinstance(observation, BfrescoxTransferObservation2):
            raise ValueError(
                f"Observation must be a BfrescoxTransferObservation2, but got {type(observation)}"
            )

        # convert params tuple to dict
        params_dict = {param.name: val for param, val in zip(self.params, params)}

        N = params_dict.pop("N")

        ANC = observation.run_single_ANC(params_dict)

        return ANC
