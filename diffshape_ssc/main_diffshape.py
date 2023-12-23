import torch
import argparse
import time
import torch.nn as nn
import numpy as np
from diffusion.diffusion_model import VDiffusion, VSampler, DiffusionModel, UNetV0
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from semi_utils import set_seed, build_loss, evaluate_model_acc, lan_shapelet_contrastive_loss, get_all_text_labels, \
    get_each_sample_distance_shapelet, get_similarity_shapelet, build_model, build_dataset, \
    get_all_datasets, fill_nan_value, normalize_per_series, shuffler, UCRDataset, \
    get_pesudo_via_high_confidence_softlabels
from semi_backbone import ProjectionHead
from parameter_shapelets import *

ucr_datasets_dict = {
    'AllGestureWiimoteX': {'0': 'poteg - pick-up', '1': 'shake - shake', '2': 'desno - one move to the right',
                           '3': 'levo - one move to the left', '4': 'gor - one move to up',
                           '5': 'dol - one move to down', '6': 'kroglevo - one left circle',
                           '7': 'krogdesno - one right circle', '8': 'suneknot - one move toward the screen',
                           '9': 'sunekven - one move away from the screen'},
    'AllGestureWiimoteY': {'0': 'poteg - pick-up', '1': 'shake - shake', '2': 'desno - one move to the right',
                           '3': 'levo - one move to the left', '4': 'gor - one move to up',
                           '5': 'dol - one move to down', '6': 'kroglevo - one left circle',
                           '7': 'krogdesno - one right circle', '8': 'suneknot - one move toward the screen',
                           '9': 'sunekven - one move away from the screen'},
    'AllGestureWiimoteZ': {'0': 'poteg - pick-up', '1': 'shake - shake', '2': 'desno - one move to the right',
                           '3': 'levo - one move to the left', '4': 'gor - one move to up',
                           '5': 'dol - one move to down', '6': 'kroglevo - one left circle',
                           '7': 'krogdesno - one right circle', '8': 'suneknot - one move toward the screen',
                           '9': 'sunekven - one move away from the screen'},
    'ArrowHead': {'0': 'Avonlea', '1': 'Clovis', '2': 'Mix'},
    'BME': {'0': 'Begin', '1': 'Middle', '2': 'End'},
    'Car': {'0': 'Sedan', '1': 'Pickup', '2': 'Minivan', '3': 'SUV'},
    'CBF': {'0': 'Cylinder', '1': 'Bell', '2': 'Funnel'},
    'Chinatown': {'0': 'Weekend', '1': 'Weekday'},
    'CinCECGTorso': {'0': 'People 1', '1': 'People 2', '2': 'People 3', '3': 'People 4'},
    'ChlorineConcentration': {'0': 'low', '1': 'middle', '2': 'high'},
    'Computers': {'0': 'Laptop', '1': 'Desktop'},
    'CricketX': {'0': 'Cancel Call', '1': 'Dead Ball', '2': 'Four', '3': 'Last Hour', '4': 'Leg Bye', '5': 'No Ball',
                 '6': 'One Short', '7': 'Out', '8': 'Penalty Runs', '9': 'Six', '10': 'TV Replay', '11': 'Wide'},
    'CricketY': {'0': 'Cancel Call', '1': 'Dead Ball', '2': 'Four', '3': 'Last Hour', '4': 'Leg Bye', '5': 'No Ball',
                 '6': 'One Short', '7': 'Out', '8': 'Penalty Runs', '9': 'Six', '10': 'TV Replay', '11': 'Wide'},
    'CricketZ': {'0': 'Cancel Call', '1': 'Dead Ball', '2': 'Four', '3': 'Last Hour', '4': 'Leg Bye', '5': 'No Ball',
                 '6': 'One Short', '7': 'Out', '8': 'Penalty Runs', '9': 'Six', '10': 'TV Replay', '11': 'Wide'},
    'Crop': {'0': 'corn', '1': 'wheat', '2': 'dense building', '3': 'built indu', '4': 'diffuse building',
             '5': 'temporary meadow', '6': 'hardwood', '7': 'wasteland', '8': 'jachere', '9': 'soy', '10': 'water',
             '11': 'pre', '12': 'softwood', '13': 'sunflower', '14': 'sorghum', '15': 'eucalyptus', '16': 'rapeseed',
             '17': 'but drilling', '18': 'barley', '19': 'peas', '20': 'poplars', '21': 'mineral surface',
             '22': 'gravel', '23': 'lake'},
    'DiatomSizeReduction': {'0': 'Gomphonema augur', '1': 'Fragilariforma bicapitata', '2': 'Stauroneis smithii',
                            '3': 'Eunotia tenella'},
    'DistalPhalanxOutlineAgeGroup': {'0': '0-6 years old', '1': '7-12 years old', '2': '13-19 years old'},
    'DistalPhalanxOutlineCorrect': {'0': 'Correct', '1': 'Incorrect'},
    'DistalPhalanxTW': {'0': '0-6 years old', '1': '7-12 years old', '2': '13-19 years old'},
    'DodgerLoopGame': {'0': 'Sunday', '1': 'Monday', '2': 'Tuesday', '3': 'Wednesday', '4': 'Thursday', '5': 'Friday',
                       '6': 'Saturday'},
    'DodgerLoopWeekend': {'0': 'Sunday', '1': 'Monday', '2': 'Tuesday', '3': 'Wednesday', '4': 'Thursday',
                          '5': 'Friday', '6': 'Saturday'},
    'Earthquakes': {'0': 'negative', '1': 'positive'},
    'ECG200': {'0': 'Ischemia', '1': 'Normal'},
    'ECG5000': {'0': 'Normal', '1': 'R-on-T premature ventricular contraction',
                '2': 'Supraventricular premature or ectopic beat', '3': 'Premature ventricular contraction',
                '4': 'Unclassifiable beat'},
    'ECGFiveDays': {'0': '12/11/1990', '1': '17/11/1990'},
    'ElectricDevices': {'0': 'screenGroup', '1': 'dishwasher', '2': 'coldGroup', '3': 'immersionHeater', '4': 'kettle',
                        '5': 'ovenCooker', '6': 'washingMachine'},
    'EOGHorizontalSignal': {'0': 'Upper left', '1': 'Up', '2': 'Upper right', '3': 'Left', '4': 'Center', '5': 'Right',
                            '6': 'Lower left', '7': 'Down', '8': 'Lower right', '9': 'Sil',
                            '10': 'Left, up, right, down, left', '11': 'Blinking'},
    'EOGVerticalSignal': {'0': 'Upper left', '1': 'Up', '2': 'Upper right', '3': 'Left', '4': 'Center', '5': 'Right',
                          '6': 'Lower left', '7': 'Down', '8': 'Lower right', '9': 'Sil',
                          '10': 'Left, up, right, down, left', '11': 'Blinking'},
    'EthanolLevel': {'0': 'Thirty-five percent ethanol', '1': 'Thirty-eight percent ethanol',
                     '2': 'Forty percent ethanol', '3': 'Forty-five percent ethanol'},
    'FaceAll': {'0': 'Student1', '1': 'Student2', '2': 'Student3', '3': 'Student4', '4': 'Student5', '5': 'Student6',
                '6': 'Student7', '7': 'Student8', '8': 'Student9', '9': 'Student10', '10': 'Student11',
                '11': 'Student12', '12': 'Student13', '13': 'Student14'},
    'FacesUCR': {'0': 'Student1', '1': 'Student2', '2': 'Student3', '3': 'Student4', '4': 'Student5', '5': 'Student6',
                 '6': 'Student7', '7': 'Student8', '8': 'Student9', '9': 'Student10', '10': 'Student11',
                 '11': 'Student12', '12': 'Student13', '13': 'Student14'},
    'Fish': {'0': 'Chinook salmon', '1': 'Winter coho', '2': 'brown trout', '3': 'Bonneville cutthroat',
             '4': 'Colorado River cutthroat trout', '5': 'Yellowstone cutthroat', '6': 'Mountain whitefish'},
    'FordA': {'0': 'not exists a certain symptom', '1': 'exists a certain symptom'},
    'FordB': {'0': 'not exists a certain symptom', '1': 'exists a certain symptom'},
    'FreezerRegularTrain': {'0': 'in the kitchen', '1': 'in the garage'},
    'FreezerSmallTrain': {'0': 'in the kitchen', '1': 'in the garage'},
    'GesturePebbleZ1': {'0': 'hh', '1': 'hu', '2': 'ud', '3': 'hud', '4': 'hh2', '5': 'hu2'},
    'GesturePebbleZ2': {'0': 'hh', '1': 'hu', '2': 'ud', '3': 'hud', '4': 'hh2', '5': 'hu2'},
    'GunPoint': {'0': 'Gun-Draw', '1': 'No gun pointing'},
    'GunPointAgeSpan': {'0': 'Gun (FG03, MG03, FG18, MG18)', '1': 'Point (FP03, MP03, FP18, MP18)'},
    'GunPointMaleVersusFemale': {'0': 'Female (FG03, FP03, FG18, FP18)', '1': 'Male (MG03, MP03, MG18, MP18)'},
    'GunPointOldVersusYoung': {'0': 'Young (FG03, MG03, FP03, MP03)', '1': 'Old (FG18, MG18, FP18, MP18)'},
    'Ham': {'0': 'Spanish', '1': 'French'},
    'HandOutlines': {'0': 'Male', '1': 'Female'},
    'Haptics': {'0': 'Person 1', '1': 'Person 2', '2': 'Person 3', '3': 'Person 4', '4': 'Person 5'},
    'Herring': {'0': 'North sea', '1': 'Thames'},
    'HouseTwenty': {'0': 'household aggregate usage of electricity',
                    '1': 'aggregate electricity load of Tumble Dryer and Washing Machine'},
    'InlineSkate': {'0': 'Individual1', '1': 'Individual2', '2': 'Individual3', '3': 'Individual4', '4': 'Individual5',
                    '5': 'Individual6', '6': 'Individual7'},
    'InsectEPGRegularTrain': {'0': 'Phloem Salivation', '1': 'Phloem Ingestion', '2': 'Xylem Ingestion'},
    'InsectEPGSmallTrain': {'0': 'Stylet passage through plant cells1', '1': 'Contact with Phloem Tissue',
                            '2': 'Contact with Phloem Tissue'},
    'InsectWingbeatSound': {'0': 'Male Ae. aegypti', '1': 'Female Ae. aegypti', '2': 'Male Cx. tarsalis',
                            '3': 'Female Cx. tarsalis', '4': 'Male Cx. quinquefasciants',
                            '5': 'Female Cx. quinquefasciants', '6': 'Male Cx. stigmatosoma',
                            '7': 'Female Cx. stigmatosoma', '8': 'Musca domestica', '9': 'Drosophila simulans',
                            '10': 'Other insects'},
    'ItalyPowerDemand': {'0': 'Days from Oct to March', '1': 'Days from April to September'},
    'LargeKitchenAppliances': {'0': 'Washing Machine', '1': 'Tumble Dryer', '2': 'Dishwasher'},
    'Lightning2': {'0': 'class 1', '1': 'class 2'},
    'Mallat': {'0': 'case 1', '1': 'case 2', '2': 'case 3', '3': 'case 4', '4': 'case 5', '5': 'case 6', '6': 'case 7',
               '7': 'Original'},
    'Meat': {'0': 'chicken', '1': 'pork', '2': 'turkey'},
    'MedicalImages': {'0': 'brain', '1': 'spine', '2': 'heart', '3': 'liver', '4': 'adiposity', '5': 'breast',
                      '6': 'muscle', '7': 'bone', '8': 'lung', '9': 'other'},
    'MelbournePedestrian': {'0': 'Bourke Street Mall (North)', '1': 'Southern Cross Station', '2': 'New Quay',
                            '3': 'Flinders St Station Underpass', '4': 'QV Market-Elizabeth (West)',
                            '5': 'Convention/Exhibition Centre', '6': 'Chinatown-Swanston St (North)',
                            '7': 'Webb Bridge', '8': 'Tin Alley-Swanston St (West)', '9': 'Southbank'},
    'MiddlePhalanxOutlineAgeGroup': {'0': '0-6 years old', '1': '7-12 years old', '2': '13-19 years old'},
    'MiddlePhalanxOutlineCorrect': {'0': 'Correct', '1': 'Incorrect'},
    'MiddlePhalanxTW': {'0': '0-6 years old is correct', '1': '0-6 years old is incorrect',
                        '2': '7-12 years old is correct', '3': '7-12 years old is incorrect',
                        '3': '13-19 years old is correct', '5': '13-19 years old is incorrect'},
    'MixedShapesRegularTrain': {'0': 'Arrowhead', '1': 'Butterfly', '2': 'Fish', '3': 'Seashell', '4': 'Shield'},
    'MixedShapesSmallTrain': {'0': 'Arrowhead', '1': 'Butterfly', '2': 'Fish', '3': 'Seashell', '4': 'Shield'},
    'MoteStrain': {'0': 'q8calibHumid', '1': 'q8calibHumTemp'},
    'NonInvasiveFetalECGThorax1': {'0': 'ECG signal 1', '1': 'ECG signal 2', '2': 'ECG signal 3', '3': 'ECG signal 4',
                                   '4': 'ECG signal 5', '5': 'ECG signal 6', '6': 'ECG signal 7', '7': 'ECG signal 8'
        , '8': 'ECG signal 9', '9': 'ECG signal 10', '10': 'ECG signal 11', '11': 'ECG signal 12',
                                   '12': 'ECG signal 13', '13': 'ECG signal 14', '14': 'ECG signal 15'
        , '15': 'ECG signal 16', '16': 'ECG signal 17', '17': 'ECG signal 18', '18': 'ECG signal 19',
                                   '19': 'ECG signal 20', '20': 'ECG signal 21', '21': 'ECG signal 22'
        , '22': 'ECG signal 23', '23': 'ECG signal 24', '24': 'ECG signal 25', '25': 'ECG signal 26',
                                   '26': 'ECG signal 27', '27': 'ECG signal 28', '28': 'ECG signal 29'
        , '29': 'ECG signal 30', '30': 'ECG signal 31', '31': 'ECG signal 32', '32': 'ECG signal 33',
                                   '33': 'ECG signal 34', '34': 'ECG signal 35', '35': 'ECG signal 36'
        , '36': 'ECG signal 37', '37': 'ECG signal 38', '38': 'ECG signal 39', '39': 'ECG signal 40',
                                   '40': 'ECG signal 41', '41': 'ECG signal 42'},
    'NonInvasiveFetalECGThorax2': {'0': 'ECG signal 1', '1': 'ECG signal 2', '2': 'ECG signal 3', '3': 'ECG signal 4',
                                   '4': 'ECG signal 5', '5': 'ECG signal 6', '6': 'ECG signal 7', '7': 'ECG signal 8'
        , '8': 'ECG signal 9', '9': 'ECG signal 10', '10': 'ECG signal 11', '11': 'ECG signal 12',
                                   '12': 'ECG signal 13', '13': 'ECG signal 14', '14': 'ECG signal 15'
        , '15': 'ECG signal 16', '16': 'ECG signal 17', '17': 'ECG signal 18', '18': 'ECG signal 19',
                                   '19': 'ECG signal 20', '20': 'ECG signal 21', '21': 'ECG signal 22'
        , '22': 'ECG signal 23', '23': 'ECG signal 24', '24': 'ECG signal 25', '25': 'ECG signal 26',
                                   '26': 'ECG signal 27', '27': 'ECG signal 28', '28': 'ECG signal 29'
        , '29': 'ECG signal 30', '30': 'ECG signal 31', '31': 'ECG signal 32', '32': 'ECG signal 33',
                                   '33': 'ECG signal 34', '34': 'ECG signal 35', '35': 'ECG signal 36'
        , '36': 'ECG signal 37', '37': 'ECG signal 38', '38': 'ECG signal 39', '39': 'ECG signal 40',
                                   '40': 'ECG signal 41', '41': 'ECG signal 42'},
    'OSULeaf': {'0': 'Acer Circinatum', '1': 'Acer Glabrum', '2': 'Acer Macrophyllum', '3': 'Acer Negundo',
                '4': 'Quercus Garryana', '5': 'Quercus Kelloggii'},
    'PhalangesOutlinesCorrect': {'0': 'Correct', '1': 'Incorrect'},
    'Phoneme': {'0': 'HH', '1': 'DH', '2': 'F', '3': 'S', '4': 'SH', '5': 'TH', '6': 'V', '7': 'Z', '8': 'ZH',
                '9': 'CH', '10': 'JH', '11': 'B', '12': 'D', '13': 'G', '14': 'K', '15': 'P', '16': 'T', '17': 'M',
                '18': 'N', '19': 'NG', '20': 'AA', '21': 'AE', '22': 'AH', '23': 'UW', '24': 'AO', '25': 'AW',
                '26': 'AY', '27': 'UH', '28': 'EH', '29': 'ER', '30': 'EY', '31': 'OY', '32': 'IH', '33': 'IY',
                '34': 'OW', '35': 'W', '36': 'Y', '37': 'L', '38': 'R'},
    'PLAID': {'0': 'air conditioner', '1': 'compact flourescent lamp', '2': 'fan', '3': 'fridge', '4': 'hairdryer',
              '5': 'heater', '6': 'incandescent light bulb', '7': 'laptop', '8': 'microwave', '9': 'vacuum',
              '10': 'wahing machine'},
    'Plane': {'0': 'Mirage', '1': 'Eurofighter', '2': 'F-14 wings closed', '3': 'F-14 wings opened', '4': 'Harrier',
              '5': 'F-22', '6': 'F-15'},
    'PowerCons': {'0': 'Warm season', '1': 'Cold season'},
    'ProximalPhalanxOutlineAgeGroup': {'0': '0-6 years old', '1': '7-12 years old', '2': '13-19 years old'},
    'ProximalPhalanxOutlineCorrect': {'0': 'Correct', '1': 'Incorrect'},
    'ProximalPhalanxTW': {'0': 'that 0-6 years old is correct', '1': 'that 0-6 years old is incorrect',
                          '2': 'that 7-12 years old is correct', '3': 'that 7-12 years old is incorrect',
                          '4': 'that 13-19 years old is correct', '5': 'that 13-19 years old is incorrect'},
    'RefrigerationDevices': {'0': 'Fridge/Freezer', '1': 'Refrigerator', '2': 'Upright Freezer'},
    'ScreenType': {'0': 'CRT TV', '1': 'LCD TV', '2': 'Computer Monitor'},
    'SemgHandGenderCh2': {'0': 'Female', '1': 'Male'},
    'SemgHandMovementCh2': {'0': 'Cylindrical', '1': 'Hook', '2': 'Tip', '3': 'Palmar', '4': 'Spherical',
                            '5': 'Lateral'},
    'SemgHandSubjectCh2': {'0': 'Female 1', '1': 'Female 2', '2': 'Female 3', '3': 'Male 1', '4': 'Male 2'},
    'ShapeletSim': {'0': 'shape 1', '1': 'shape 2'},
    'SmallKitchenAppliances': {'0': 'Kettle', '1': 'Microwave', '2': 'Toaster'},
    'SmoothSubspace': {'0': 'from time stamp 1-5 ', '1': 'from time stamp 6-10', '2': 'from time stamp 11-15'},
    'SonyAIBORobotSurface1': {'0': 'walk on carpet', '1': 'walk on cement'},
    'SonyAIBORobotSurface2': {'0': 'walk on carpet', '1': 'walk on cement'},
    'StarLightCurves': {'0': 'Cepheid', '1': 'Eclipsing Binary', '2': 'RR Lyrae'},
    'Strawberry': {'0': 'strawberry', '1': 'non-strawberry'},
    'SwedishLeaf': {'0': 'Ulmus carpinifolia', '1': 'Acer', '2': 'Salix aurita', '3': 'Quercus', '4': 'Alnus incana',
                    '5': 'Betula pubescens', '6': 'Salix alba Sericea', '7': 'Populus tremula', '8': 'Ulmus glabra',
                    '9': 'Sorbus aucuparia', '10': 'Salix sinerea', '11': 'Populus', '12': 'Tilia',
                    '13': 'Sorbus intermedia', '14': 'Fagus silvatica'},
    'Symbols': {'0': 'symbol 1', '1': 'symbol 2', '2': 'symbol 3', '3': 'symbol 4', '4': 'symbol 5', '5': 'symbol 6'},
    'SyntheticControl': {'0': 'Normal', '1': 'Cyclic', '2': 'Increasing trend', '3': 'Decreasing trend',
                         '4': 'Upward shift', '5': 'Downward shift'},
    'ToeSegmentation1': {'0': 'normal walk', '1': 'abnormal walk'},
    'ToeSegmentation2': {'0': 'normal walk', '1': 'abnormal walk'},
    'Trace': {'0': 'The second feature of class two', '1': 'The second feature of class six',
              '2': 'The third feature of class three', '3': 'The third feature of class seven'},
    'TwoLeadECG': {'0': 'signal 0', '1': 'signal 1'},
    'TwoPatterns': {'0': 'down-down', '1': 'up-down', '2': 'down-up', '3': 'up-up'},
    'UMD': {'0': 'Up', '1': 'Middle', '2': 'Down'},
    'UWaveGestureLibraryAll': {'0': 'fold line', '1': 'clockwise square', '2': 'right arrow', '3': 'left arrow',
                               '4': 'up arrow', '5': 'down arrow', '6': 'clockwise circler',
                               '7': 'anticlockwise circle'},
    'UWaveGestureLibraryX': {'0': 'fold line', '1': 'clockwise square', '2': 'right arrow', '3': 'left arrow',
                             '4': 'up arrow', '5': 'down arrow', '6': 'clockwise circler', '7': 'anticlockwise circle'},
    'UWaveGestureLibraryY': {'0': 'fold line', '1': 'clockwise square', '2': 'right arrow', '3': 'left arrow',
                             '4': 'up arrow', '5': 'down arrow', '6': 'clockwise circler', '7': 'anticlockwise circle'},
    'UWaveGestureLibraryZ': {'0': 'fold line', '1': 'clockwise square', '2': 'right arrow', '3': 'left arrow',
                             '4': 'up arrow', '5': 'down arrow', '6': 'clockwise circler', '7': 'anticlockwise circle'},
    'Wafer': {'0': 'normal', '1': 'abnormal'},
    'Wine': {'0': 'strawberry', '1': 'non-strawberry'},
    'WordSynonyms': {'0': 'one', '1': 'two', '2': 'three', '3': 'four', '4': 'five', '5': 'six', '6': 'seven',
                     '7': 'eight', '8': 'nine', '9': 'ten', '10': 'eleven', '11': 'twelve', '12': 'thirteen',
                     '13': 'fourteen', '14': 'fifteen', '15': 'sixteen', '16': 'seventeen', '17': 'eighteen',
                     '18': 'nineteen', '19': 'twenty', '20': 'twenty-one', '21': 'twenty-two', '22': 'twenty-three',
                     '23': 'twenty-four', '24': 'twenty-five'},
    'Worms': {'0': 'wild-type', '1': 'goa-1', '2': 'unc-1', '3': 'unc-38', '4': 'unc-63'},
    'WormsTwoClass': {'0': 'wild-type', '1': 'mutant'},
    'Yoga': {'0': 'male', '1': 'female'},
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Base setup
    parser.add_argument('--backbone', type=str, default='fcn', help='encoder backbone, fcn')
    parser.add_argument('--random_seed', type=int, default=42, help='shuffle seed')
    parser.add_argument('--total_dim', type=int, default=100, help='total numbers of slicing shapelets')
    parser.add_argument('--target_dim', type=int, default=5, help='2, 5, 10')
    parser.add_argument('--len_shapelet_ratio', type=float, default=0.2, help='0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8')

    # Dataset setup
    parser.add_argument('--dataset', type=str, default='Trace', help='')
    parser.add_argument('--dataroot', type=str, default='.../UCRArchive_2018', help='path of UCR folder')
    parser.add_argument('--num_classes', type=int, default=2, help='number of class')
    parser.add_argument('--input_size', type=int, default=2, help='input_size')

    # training setup
    parser.add_argument('--labeled_ratio', type=float, default=0.1, help='0.1, 0.2, 0.4')
    parser.add_argument('--warmup_epochs', type=int, default=300,
                        help='warmup epochs using only labeled data for ssl')
    parser.add_argument('--temperature', type=float, default=50, help='20 or 50')
    parser.add_argument('--sup_con_mu', type=float, default=0.001, help='0.001 or 0.005')  ## prompt_toolkit_series_i
    parser.add_argument('--sup_df', type=float, default=0.01,
                        help='0.001 or 0.01')
    parser.add_argument('--prompt_toolkit_series_i', type=int, default=0, help='')

    parser.add_argument('--loss', type=str, default='cross_entropy', help='loss function')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='')
    parser.add_argument('--epoch', type=int, default=1000, help='training epoch')
    parser.add_argument('--cuda', type=str, default='cuda:0')

    # classifier setup
    parser.add_argument('--classifier', type=str, default='linear', help='')
    parser.add_argument('--classifier_input', type=int, default=128, help='input dim of the classifiers')

    args = parser.parse_args()
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    set_seed(args)

    if args.labeled_ratio == 0.1:
        args.total_dim = ucr_hyp_dict_10[args.dataset]["target_dim"]
        args.len_shapelet_ratio = ucr_hyp_dict_10[args.dataset]["len_shapelet_ratio"]

    if args.labeled_ratio == 0.2:
        args.total_dim = ucr_hyp_dict_20[args.dataset]["target_dim"]
        args.len_shapelet_ratio = ucr_hyp_dict_20[args.dataset]["len_shapelet_ratio"]

    if args.labeled_ratio == 0.4:
        args.total_dim = ucr_hyp_dict_40[args.dataset]["target_dim"]
        args.len_shapelet_ratio = ucr_hyp_dict_40[args.dataset]["len_shapelet_ratio"]

    sum_dataset, sum_target, num_classes = build_dataset(args)
    args.num_classes = num_classes

    train_datasets, train_targets, val_datasets, val_targets, test_datasets, test_targets = get_all_datasets(
        sum_dataset, sum_target)

    len_series = train_datasets[0].shape[-1]

    window_shape_s = np.int(len_series * args.len_shapelet_ratio)
    temp_train_dataset = np.lib.stride_tricks.sliding_window_view(
        x=train_datasets[0],
        window_shape=window_shape_s,
        axis=1
    )
    args.input_size = args.target_dim
    args.total_dim = temp_train_dataset.shape[1]

    while args.batch_size * 2 > train_datasets[0].shape[0]:
        args.batch_size = args.batch_size // 2

    text_embedding_labels = get_all_text_labels(ucr_datasets_dict=ucr_datasets_dict,
                                                dataset_name=args.dataset, num_labels=num_classes, device=device,
                                                prompt_toolkit_series_i=args.prompt_toolkit_series_i)

    fcn_model, fcn_classifier = build_model(args)
    fcn_model = fcn_model.to(device)
    fcn_classifier = fcn_classifier.to(device)

    mlp_text_head = ProjectionHead(input_dim=768, output_dim=128).to(device)

    conv_model = nn.Conv2d(in_channels=args.total_dim, out_channels=args.target_dim, kernel_size=3, padding='same')
    conv_model = conv_model.to(device)

    loss_fcn = build_loss(args).to(device)

    df_model = DiffusionModel(
        net_t=UNetV0,
        in_channels=args.target_dim,
        channels=[8, 32, 64, 64],
        factors=[1, 1, 1, 1],
        items=[1, 2, 2, 2],
        attentions=[0, 0, 0, 1],
        cross_attentions=[0, 0, 0, 1],
        modulation_features=64,
        attention_heads=8,
        attention_features=64,
        diffusion_t=VDiffusion,
        sampler_t=VSampler,
        use_embedding_cfg=True,
        embedding_features=window_shape_s,
        embedding_max_length=args.target_dim,
    )
    df_model = df_model.to(device)

    conv_model_init_state = conv_model.state_dict()
    fcn_model_init_state = fcn_model.state_dict()
    fcn_classifier_init_state = fcn_classifier.state_dict()
    mlp_text_head_init_state = mlp_text_head.state_dict()
    df_model_init_state = df_model.state_dict()

    optimizer = torch.optim.Adam([{'params': conv_model.parameters()}, {'params': mlp_text_head.parameters()},
                                  {'params': df_model.parameters()}, {'params': fcn_model.parameters()},
                                  {'params': fcn_classifier.parameters()}],
                                 lr=args.lr)

    print('Start training on {}'.format(args.dataset))

    losses = []
    test_accuracies = []
    train_time = 0.0
    end_val_epochs = []

    for i, train_dataset in enumerate(train_datasets):
        t = time.time()

        conv_model.load_state_dict(conv_model_init_state)
        fcn_model.load_state_dict(fcn_model_init_state)
        fcn_classifier.load_state_dict(fcn_classifier_init_state)
        df_model.load_state_dict(df_model_init_state)
        mlp_text_head.load_state_dict(mlp_text_head_init_state)

        print('{} fold start training and evaluate'.format(i))

        train_target = train_targets[i]
        val_dataset = val_datasets[i]
        val_target = val_targets[i]

        test_dataset = test_datasets[i]
        test_target = test_targets[i]

        train_dataset, val_dataset, test_dataset = fill_nan_value(train_dataset, val_dataset, test_dataset)

        # TODO normalize per series
        train_dataset = normalize_per_series(train_dataset)
        val_dataset = normalize_per_series(val_dataset)
        test_dataset = normalize_per_series(test_dataset)

        train_dataset = np.lib.stride_tricks.sliding_window_view(
            x=train_dataset,
            window_shape=window_shape_s,
            axis=1
        )

        val_dataset = np.lib.stride_tricks.sliding_window_view(
            x=val_dataset,
            window_shape=window_shape_s,
            axis=1
        )

        test_dataset = np.lib.stride_tricks.sliding_window_view(
            x=test_dataset,
            window_shape=window_shape_s,
            axis=1
        )

        train_labeled, train_unlabeled, y_labeled, y_unlabeled = train_test_split(train_dataset, train_target,
                                                                                  test_size=(
                                                                                          1 - args.labeled_ratio),
                                                                                  random_state=args.random_seed)
        mask_labeled = np.zeros(len(y_labeled))
        mask_unlabeled = np.ones(len(y_unlabeled))
        mask_train = np.concatenate([mask_labeled, mask_unlabeled])
        train_all_split = np.concatenate([train_labeled, train_unlabeled])
        y_label_split = np.concatenate([y_labeled, y_unlabeled])

        x_train_all, y_train_all = shuffler(train_all_split, y_label_split)
        mask_train, _ = shuffler(mask_train, mask_train)
        y_train_all[mask_train == 1] = -1  ## Generate unlabeled data

        x_train_all = torch.from_numpy(x_train_all).to(device)
        y_train_all = torch.from_numpy(y_train_all).to(device).to(torch.int64)

        x_train_labeled_all = x_train_all[mask_train == 0]
        y_train_labeled_all = y_train_all[mask_train == 0]

        train_set_labled = UCRDataset(x_train_labeled_all, y_train_labeled_all)
        train_set = UCRDataset(x_train_all, y_train_all)
        val_set = UCRDataset(torch.from_numpy(val_dataset).to(device),
                             torch.from_numpy(val_target).to(device).to(torch.int64))
        test_set = UCRDataset(torch.from_numpy(test_dataset).to(device),
                              torch.from_numpy(test_target).to(device).to(torch.int64))

        train_labeled_loader = DataLoader(train_set_labled, batch_size=args.batch_size, num_workers=0,
                                          drop_last=False)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=0, drop_last=False)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0)

        last_val_accu = 0
        stop_count = 0
        increase_count = 0

        num_steps = train_set.__len__() // args.batch_size

        max_val_accu = 0
        test_accuracy = 0
        end_val_epoch = 0

        for epoch in range(args.epoch):

            if stop_count == 80 or increase_count == 80:
                print('model convergent at epoch {}, early stopping'.format(epoch))
                break

            num_iterations = 0
            epoch_train_loss = 0

            conv_model.train()
            df_model.train()
            fcn_model.train()
            fcn_classifier.train()
            mlp_text_head.train()

            if epoch < args.warmup_epochs:
                for x, y in train_labeled_loader:

                    if x.shape[0] < 2:
                        continue

                    optimizer.zero_grad()
                    predicted = conv_model(torch.unsqueeze(x, 2))

                    raw_similar_shapelets_list = None
                    transformation_similart_loss = None
                    for _i in range(predicted.shape[0]):
                        _, _raw_similar_shapelets = get_each_sample_distance_shapelet(
                            generator_shapelet=predicted[_i],
                            raw_shapelet=x[_i],
                            topk=1)
                        _raw_similar_shapelets = torch.unsqueeze(_raw_similar_shapelets, 0)
                        if raw_similar_shapelets_list is None:
                            raw_similar_shapelets_list = _raw_similar_shapelets
                        else:
                            raw_similar_shapelets_list = torch.cat((raw_similar_shapelets_list, _raw_similar_shapelets),
                                                                   0)

                        _i_sim_loss = get_similarity_shapelet(generator_shapelet=predicted[_i])
                        if transformation_similart_loss == None:
                            transformation_similart_loss = 0.01 * _i_sim_loss
                        else:
                            transformation_similart_loss = transformation_similart_loss + 0.01 * _i_sim_loss

                    loss_df = df_model(torch.squeeze(predicted, 2), embedding=raw_similar_shapelets_list)

                    fcn_cls_emb = fcn_model(torch.squeeze(predicted, 2))

                    text_embedding = mlp_text_head(text_embedding_labels)
                    text_embd_batch = None
                    _i = 0
                    for _y in y:
                        temp_text_embd = torch.unsqueeze(text_embedding[_y], 0)
                        if text_embd_batch is None:
                            text_embd_batch = temp_text_embd
                        else:
                            text_embd_batch = torch.cat((text_embd_batch, temp_text_embd), 0)
                        _i = _i + 1

                    batch_sup_contrastive_loss = lan_shapelet_contrastive_loss(
                        embd_batch=torch.nn.functional.normalize(fcn_cls_emb),
                        text_embd_batch=torch.nn.functional.normalize(text_embd_batch),
                        labels=y,
                        device=device,
                        temperature=args.temperature,
                        base_temperature=args.temperature)

                    fcn_cls_prd = fcn_classifier(fcn_cls_emb)
                    step_loss_fcn = loss_fcn(fcn_cls_prd, y)

                    sum_loss = transformation_similart_loss + args.sup_df * loss_df + step_loss_fcn + batch_sup_contrastive_loss * args.sup_con_mu  ##

                    sum_loss.backward()
                    optimizer.step()

                    epoch_train_loss += sum_loss.item()
            else:
                for x, y in train_loader:

                    if (num_iterations + 1) * args.batch_size < train_set.__len__():
                        mask_train_batch = mask_train[
                                           num_iterations * args.batch_size: (num_iterations + 1) * args.batch_size]
                    else:
                        mask_train_batch = mask_train[num_iterations * args.batch_size:]

                    mask_labeled = [True if mask_train_batch[m] == 0 else False for m in range(len(mask_train_batch))]

                    optimizer.zero_grad()

                    predicted = conv_model(torch.unsqueeze(x, 2))

                    raw_similar_shapelets_list = None
                    transformation_similart_loss = None
                    for _i in range(predicted.shape[0]):
                        _, _raw_similar_shapelets = get_each_sample_distance_shapelet(
                            generator_shapelet=predicted[_i],
                            raw_shapelet=x[_i],
                            topk=1)
                        _raw_similar_shapelets = torch.unsqueeze(_raw_similar_shapelets, 0)
                        if raw_similar_shapelets_list is None:
                            raw_similar_shapelets_list = _raw_similar_shapelets
                        else:
                            raw_similar_shapelets_list = torch.cat((raw_similar_shapelets_list, _raw_similar_shapelets),
                                                                   0)

                        _i_sim_loss = get_similarity_shapelet(generator_shapelet=predicted[_i])
                        if transformation_similart_loss == None:
                            transformation_similart_loss = _i_sim_loss * 0.01
                        else:
                            transformation_similart_loss = transformation_similart_loss + _i_sim_loss * 0.01

                    loss_df = df_model(torch.squeeze(predicted, 2), embedding=raw_similar_shapelets_list)

                    fcn_cls_emb = fcn_model(torch.squeeze(predicted, 2))

                    new_mask_labeled = None
                    end_all_label = None
                    if len(y[mask_labeled]) >= 1:

                        fcn_cls_prd = fcn_classifier(fcn_cls_emb)

                        if epoch > args.warmup_epochs:

                            new_mask_labeled, end_all_label = get_pesudo_via_high_confidence_softlabels(y_label=y,
                                                                                                        pseudo_label_soft=fcn_cls_prd,
                                                                                                        mask_label=mask_labeled,
                                                                                                        num_real_class=args.num_classes,
                                                                                                        device=device,
                                                                                                        p_cutoff=0.95)

                            step_loss_fcn = loss_fcn(fcn_cls_prd[new_mask_labeled], end_all_label[new_mask_labeled])
                        else:
                            step_loss_fcn = loss_fcn(fcn_cls_prd[mask_labeled], y[mask_labeled])

                    else:
                        step_loss_fcn = 0

                    if len(y[mask_labeled]) >= 1:
                        if new_mask_labeled is not None:
                            _mask_labeled = new_mask_labeled
                            y = end_all_label
                        else:
                            _mask_labeled = mask_labeled

                        text_embedding = mlp_text_head(text_embedding_labels)
                        text_embd_batch = None
                        _i = 0
                        for _y in y[_mask_labeled]:
                            temp_text_embd = torch.unsqueeze(text_embedding[_y], 0)
                            if text_embd_batch is None:
                                text_embd_batch = temp_text_embd
                            else:
                                text_embd_batch = torch.cat((text_embd_batch, temp_text_embd), 0)
                            _i = _i + 1

                        batch_sup_contrastive_loss = lan_shapelet_contrastive_loss(
                            embd_batch=torch.nn.functional.normalize(fcn_cls_emb[_mask_labeled]),
                            text_embd_batch=torch.nn.functional.normalize(text_embd_batch),
                            labels=y[_mask_labeled],
                            device=device,
                            temperature=args.temperature,
                            base_temperature=args.temperature)
                    else:
                        batch_sup_contrastive_loss = 0

                    step_loss2 = 0.0
                    if epoch > args.warmup_epochs:
                        _a, _b, _c = torch.squeeze(predicted[mask_labeled], 2).shape

                        noise = torch.randn(_a, _b, _c).to(device)  # [batch_size, in_channels, length]
                        df_sample = df_model.sample(noise, num_steps=10,
                                                    x_raw_s=raw_similar_shapelets_list[mask_labeled],
                                                    embedding=raw_similar_shapelets_list[
                                                        mask_labeled])  # Suggested num_steps 5, 10, 20
                        df_sample = torch.unsqueeze(df_sample, 2)
                        fcn_cls_emb1 = fcn_model(torch.squeeze(df_sample, 2))
                        fcn_cls_prd1 = fcn_classifier(fcn_cls_emb1)
                        step_loss_fcn2 = loss_fcn(fcn_cls_prd1, y[mask_labeled])
                        step_loss2 = step_loss_fcn2

                    sum_loss = transformation_similart_loss + args.sup_df * (
                        loss_df) + step_loss_fcn + batch_sup_contrastive_loss * args.sup_con_mu + step_loss2  ##

                    sum_loss.backward()
                    optimizer.step()

                    epoch_train_loss += sum_loss.item()

                    num_iterations += 1

            conv_model.eval()
            df_model.eval()
            fcn_model.eval()
            fcn_classifier.eval()
            mlp_text_head.eval()

            val_accu = evaluate_model_acc(val_loader, conv_model, fcn_model, fcn_classifier)

            if max_val_accu < val_accu:
                max_val_accu = val_accu
                end_val_epoch = epoch
                test_accuracy = evaluate_model_acc(test_loader, conv_model, fcn_model, fcn_classifier)

            if (epoch > args.warmup_epochs) and (last_val_accu >= val_accu):
                stop_count += 1
            else:
                stop_count = 0

            if (epoch > args.warmup_epochs) and (end_val_epoch + 80 < epoch):
                increase_count += 1
            else:
                increase_count = 0

            last_val_accu = val_accu

            if epoch % 50 == 0:
                print("epoch : {}, train loss: {}, val_accu: {}, stop_count: {}, test_accuracy: {}".format(epoch,
                                                                                                           epoch_train_loss,
                                                                                                           val_accu,
                                                                                                           stop_count,
                                                                                                           test_accuracy))

        test_accuracies.append(test_accuracy)
        train_time = time.time() - t

    test_acc_list = test_accuracies
    train_time_end = train_time
    print("The average test acc = ", np.mean(test_acc_list), ", training time = ", train_time_end)
    print('Done!')
