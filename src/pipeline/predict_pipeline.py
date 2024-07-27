import sys
import os 

import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object,load_selected_features


class PredictPipeline:
    def __init__(self):
        self.preprocessor_path = os.path.join("artifacts","preprocessor.p")
        self.model_path = os.path.join("artifacts","model.p")

    def predict(self,input_data):

        try : 
            preprocessor = load_object(path=self.preprocessor_path)
            logging.info("Preprocessor loaded in Prediction Pipeline")
            model = load_object(path=self.model_path)
            logging.info("Model loaded in Prediction Pipeline")
            input_arr = preprocessor.transform(input_data)
            logging.info("Input Data preprocessed")
            prediction = model.predict(input_arr)[0]
            logging.info("Model Prediction done")
            return prediction

        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,
                 area:str,
                 year:int,
                 savanna_fires:float,
                 forest_fires:float,
                 crop_residues:float,
                 rice_cultivation:float,
                 drained_organic_soils_C02:float,
                 pesticides_manufacturing:float,
                 food_transport:float,
                 forest_land:float,
                 net_forest_converstion:float,
                 food_household_consumption:float,
                 food_retail:float,
                 on_farm_electricity_use:float,
                 food_packaging:float,
                 agrifood_system_waste_disposal:float,
                 food_processing:float,
                 fertilizers_manufacturing:float,
                 manure_applied_to_soils:float,
                 manure_left_on_pasture:float,
                 fires_in_organic_soils:float,
                 fires_in_humid_tropical_forest:float,
                 on_farm_energy_use:float,
                 rural_population:int,
                 urban_population:int):
        self.area = area
        self.year = year
        self.savanna_fires = savanna_fires
        self.forest_fires = forest_fires
        self.crop_residues = crop_residues
        self.rice_cultivation = rice_cultivation
        self.drained_organic_soils_C02 = drained_organic_soils_C02
        self.pesticides_manufacturing = pesticides_manufacturing
        self.food_transport = food_transport
        self.forest_land = forest_land
        self.net_forest_converstion = net_forest_converstion
        self.food_household_consumption = food_household_consumption
        self.food_retail = food_retail
        self.on_farm_electricity_use = on_farm_electricity_use
        self.food_packaging = food_packaging
        self.agrifood_system_waste_disposal = agrifood_system_waste_disposal
        self.food_processing = food_processing
        self.fertilizers_manufacturing = fertilizers_manufacturing
        self.manure_applied_to_soils = manure_applied_to_soils
        self.manure_left_on_pasture = manure_left_on_pasture
        self.fires_in_organic_soils = fires_in_organic_soils
        self.fires_in_humid_tropical_forest = fires_in_humid_tropical_forest
        self.on_farm_energy_use = on_farm_energy_use
        self.rural_population = rural_population
        self.urban_population = urban_population

    def get_data_as_data_frame(self):

        try: 
            custom_data_input_array = np.array([[
                self.area,self.year,self.savanna_fires,self.forest_fires,self.crop_residues,
                self.rice_cultivation,self.drained_organic_soils_C02,self.pesticides_manufacturing,
                self.food_transport,self.forest_land,self.net_forest_converstion,
                self.food_household_consumption,self.food_retail,self.on_farm_electricity_use,
                self.food_packaging,self.agrifood_system_waste_disposal,self.food_processing,
                self.fertilizers_manufacturing,self.manure_applied_to_soils,self.manure_left_on_pasture,
                self.fires_in_organic_soils,self.fires_in_humid_tropical_forest,self.on_farm_energy_use,
                self.rural_population,self.urban_population
            ]])
            selected_features_path = os.path.join("notebook","selected_features.json")
            data_fields_names = load_selected_features(selected_features_path)
            logging.info("Input Data returned as DataFrame")
            return pd.DataFrame(
                data = custom_data_input_array,
                columns = data_fields_names
                )
            
        except Exception as e: 
            raise CustomException(e,sys)