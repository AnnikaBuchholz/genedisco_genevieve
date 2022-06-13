import numpy as np
from typing import AnyStr, List
from slingpy import AbstractDataSource
from slingpy.models.abstract_base_model import AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import \
    BaseBatchAcquisitionFunction
import random
import math


class ModelPredictionAquisition(BaseBatchAcquisitionFunction):
    def __call__(self,
                 dataset_x: AbstractDataSource,
                 select_size: int,
                 available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr] = None,
                 model: AbstractBaseModel = None,
                 ) -> List:
        avail_dataset_x = dataset_x.subset(available_indices)
        model_pedictions = model.predict(avail_dataset_x, return_std_and_margin=False)

        pred_mean = model_pedictions
        print(type(pred_mean))

        if len(pred_mean) < select_size:
            selected_indices = random.choices(available_indices, k=select_size)
            print("Error 1, picked random choice.")
            return selected_indices

        pred_mean_absolute_values = np.abs(pred_mean)
        best_hits_indices = np.argsort(pred_mean_absolute_values)[-select_size * 2:]

        selected_indices = random.choices(best_hits_indices, k=math.ceil(select_size / 2)) + random.choices(
            available_indices, k=select_size - math.ceil(select_size / 2))
        if len(selected_indices) != select_size:
            selected_indices = random.choices(available_indices, k=select_size)
            print("Error 2, picked random choice.")

        return selected_indices
