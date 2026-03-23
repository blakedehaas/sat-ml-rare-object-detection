import json
import logging
from pathlib import Path
import yaml
from tqdm import tqdm
from ultralytics import YOLO

# -------------------------------------------------------------------------
# Robust Debugging and Logging Configuration
# -------------------------------------------------------------------------
logging_instance = logging.getLogger("CoordinateSearchOptimizer")
logging_instance.setLevel(logging.DEBUG)

console_output_handler = logging.StreamHandler()
console_output_handler.setLevel(logging.INFO)

logging_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_output_handler.setFormatter(logging_formatter)
logging_instance.addHandler(console_output_handler)


# -------------------------------------------------------------------------
# Object-Oriented Component Definitions
# -------------------------------------------------------------------------
class HyperparameterSearchConfiguration:
    """
    Defines the boundaries and activation status for a single hyperparameter in the search space.
    """
    def __init__(self, parameter_name: str, is_enabled: bool, values_to_test: list[float]):
        self.parameter_name = parameter_name
        self.is_enabled = is_enabled
        self.values_to_test = values_to_test


class ProgressStateManager:
    """
    Manages saving and loading the optimization progress to disk to allow for crash recovery.
    """
    def __init__(self, progress_file_path: Path):
        self.progress_file_path = progress_file_path
        self.progress_data_dictionary = self._load_progress_from_disk()

    def _load_progress_from_disk(self) -> dict:
        if self.progress_file_path.exists():
            with open(self.progress_file_path, 'r') as file_descriptor:
                return json.load(file_descriptor)
        return {
            "highest_evaluation_metric": 0.0,
            "evaluated_parameter_combinations": {},
            "optimal_parameters_found": {}
        }

    def save_progress_to_disk(self):
        with open(self.progress_file_path, 'w') as file_descriptor:
            json.dump(self.progress_data_dictionary, file_descriptor, indent=4)

    def has_combination_been_evaluated(self, parameter_name: str, test_value: float) -> bool:
        combination_identifier = f"{parameter_name}_{test_value}"
        return combination_identifier in self.progress_data_dictionary["evaluated_parameter_combinations"]

    def record_evaluation_score(self, parameter_name: str, test_value: float, evaluation_score: float):
        combination_identifier = f"{parameter_name}_{test_value}"
        self.progress_data_dictionary["evaluated_parameter_combinations"][combination_identifier] = evaluation_score
        self.save_progress_to_disk()

    def register_new_optimal_score(self, new_score: float, complete_parameter_set: dict) -> bool:
        if new_score > self.progress_data_dictionary["highest_evaluation_metric"]:
            self.progress_data_dictionary["highest_evaluation_metric"] = new_score
            self.progress_data_dictionary["optimal_parameters_found"] = complete_parameter_set.copy()
            self.save_progress_to_disk()
            return True
        return False


class CoordinateSearchOptimizer:
    """
    Orchestrates the sequential coordinate search, evaluating one hyperparameter at a time.
    """
    def __init__(self, 
                 initial_arguments_file_path: Path, 
                 optimized_arguments_file_path: Path, 
                 hyperparameter_search_space: list):
        
        self.initial_arguments_file_path = initial_arguments_file_path
        self.optimized_arguments_file_path = optimized_arguments_file_path
        self.hyperparameter_search_space = hyperparameter_search_space
        
        self.progress_state_manager = ProgressStateManager(Path("coordinate_search_progress.json"))
        self.current_working_parameters = self._read_yaml_file(self.initial_arguments_file_path)

        # Reapply any optimal parameters discovered in previous interrupted sessions
        if self.progress_state_manager.progress_data_dictionary["optimal_parameters_found"]:
            for parameter_key, parameter_value in self.progress_state_manager.progress_data_dictionary["optimal_parameters_found"].items():
                self.current_working_parameters[parameter_key] = parameter_value
            self._write_yaml_file(self.current_working_parameters, self.optimized_arguments_file_path)

    def _read_yaml_file(self, file_path: Path) -> dict:
        with open(file_path, 'r') as file_descriptor:
            return yaml.safe_load(file_descriptor)

    def _write_yaml_file(self, data_dictionary: dict, file_path: Path):
        with open(file_path, 'w') as file_descriptor:
            yaml.dump(data_dictionary, file_descriptor, default_flow_style=False)

    def evaluate_model_performance(self, temporary_training_parameters: dict) -> float:
        """
        Trains the neural network and returns a weighted fitness metric prioritizing rare classes.
        """
        # --- CONFIGURATION FOR RARE OBJECTS ---
        # Replace these with the actual class indices from your dataset (less than 500 instances)
        #  17: Railway Vehicle
        #  36: Reach Stacker
        #  37: Straddle Carrier
        #  3: Helicopter
        #  41: Scraper/Tractor
        #  45: Ground Grader
        #  0: Fixed-wing Aircraft
        #  59: Tower
        #  21: Tank car
        #  32: Oil Tanker
        #  22: Locomotive
        #  54: Helipad
        #  20: Flat Car
        #  15: Truck Tractor w/ Liquid Tank
        #  16: Crane Truck
        #  34: Tower crane
        #  33: Engineering Vehicle
        #  27: Barge
        #  29: Ferry
        #  35: Container Crane
        #  26: Tugboat
        #  49: Aircraft Hangar
        #  44: Cement Mixer
        #  40: Haul Truck
        #  1: Small Aircraft
        #  38: Mobile Crane
        #  58: Pylon
        #  31: Container Ship
        #  30: Yacht
        #  42: Front loader/Bulldozer
        #  25: Sailboat
        #  46: Hut/Tent
        #  23: Maritime Vessel
        #  28: Fishing Vessel
        #  2: Passenger/Cargo Plane
        #  12: Truck Tractor
        #  43: Excavator
        #  14: Truck Tractor w/ Flatbed Trailer


        RARE_CLASS_INDICES = [
            17, 36, 37, 3, 41, 45, 0, 59, 21, 32, 22, 54, 20, 15, 16, 34,
            33, 27, 29, 35, 26, 49, 44, 40, 1, 38, 58, 31, 30, 42,
            25, 46, 23, 28, 2, 12, 43, 14
            ]
        RARE_WEIGHT = 0.75  # 75% focus on rare classes, 25% on global performance
        # --------------------------------------

        temporary_arguments_path = Path("temporary_training_arguments.yaml")
        self._write_yaml_file(temporary_training_parameters, temporary_arguments_path)

        logging_instance.debug("Initializing YOLO model for evaluation phase.")
        
        try:
            model_architecture_path = temporary_training_parameters.get("model", "yolo26s.pt")
            you_only_look_once_model = YOLO(model_architecture_path)
            
            # Execute training
            training_results = you_only_look_once_model.train(**temporary_training_parameters)
            
            # 1. Global mAP50 (The standard stability metric)
            global_map50 = training_results.results_dict.get('metrics/mAP50(B)', 0.0)
            
            # 2. Per-Class mAP50
            # training_results.ap50 is a list/array of AP scores at IoU=0.5 for every class
            per_class_ap50 = training_results.ap50 
            
            # Extract scores for our rare classes
            rare_class_scores = [per_class_ap50[i] for i in RARE_CLASS_INDICES if i < len(per_class_ap50)]
            
            if rare_class_scores:
                avg_rare_ap50 = sum(rare_class_scores) / len(rare_class_scores)
            else:
                avg_rare_ap50 = 0.0
                logging_instance.warning("Rare class indices not found in results. Check your RARE_CLASS_INDICES.")

            # 3. Calculate Final Weighted Fitness
            weighted_fitness = (RARE_WEIGHT * avg_rare_ap50) + ((1.0 - RARE_WEIGHT) * global_map50)
            
            logging_instance.info(
                f"Evaluation Result -> Global mAP50: {global_map50:.4f} | "
                f"Rare Avg AP50: {avg_rare_ap50:.4f} | "
                f"Weighted Fitness: {weighted_fitness:.4f}"
            )
            
            return float(weighted_fitness)
            
        except Exception as runtime_exception:
            logging_instance.error(f"Model training and evaluation encountered a failure: {runtime_exception}")
            return 0.0

    def execute_optimization_sequence(self):
        """
        Runs the primary loop, evaluating parameters and saving the highest scoring combinations.
        """
        logging_instance.info("Commencing coordinate search optimization sequence...")

        for hyperparameter_configuration in tqdm(self.hyperparameter_search_space, desc="Optimizing Hyperparameters"):
            if not hyperparameter_configuration.is_enabled:
                logging_instance.info(f"Skipping {hyperparameter_configuration.parameter_name} because it is disabled in the configuration.")
                continue

            logging_instance.info(f"Targeting parameter: {hyperparameter_configuration.parameter_name}")
            
            best_value_for_current_parameter = self.current_working_parameters.get(hyperparameter_configuration.parameter_name)
            highest_metric_for_current_parameter = self.progress_state_manager.progress_data_dictionary["highest_evaluation_metric"]

            for test_value in tqdm(hyperparameter_configuration.values_to_test, desc=f"Evaluating values for {hyperparameter_configuration.parameter_name}", leave=False):
                
                if self.progress_state_manager.has_combination_been_evaluated(hyperparameter_configuration.parameter_name, test_value):
                    logging_instance.debug(f"Skipping redundant evaluation for {hyperparameter_configuration.parameter_name} at value {test_value}")
                    continue

                test_parameters_dictionary = self.current_working_parameters.copy()
                test_parameters_dictionary[hyperparameter_configuration.parameter_name] = test_value

                logging_instance.info(f"Testing {hyperparameter_configuration.parameter_name} at value: {test_value}")
                evaluation_score = self.evaluate_model_performance(test_parameters_dictionary)

                self.progress_state_manager.record_evaluation_score(hyperparameter_configuration.parameter_name, test_value, evaluation_score)

                if evaluation_score > highest_metric_for_current_parameter:
                    highest_metric_for_current_parameter = evaluation_score
                    best_value_for_current_parameter = test_value
                    logging_instance.info(f"Discovered new local optimal value for {hyperparameter_configuration.parameter_name}: {test_value} resulting in score {evaluation_score}")

                    if self.progress_state_manager.register_new_optimal_score(evaluation_score, test_parameters_dictionary):
                        logging_instance.info(f"*** Achieved New Global Maximum Metric Score: {evaluation_score} ***")

            # Finalize the optimal value for the current parameter before moving to the next coordinate
            self.current_working_parameters[hyperparameter_configuration.parameter_name] = best_value_for_current_parameter
            self._write_yaml_file(self.current_working_parameters, self.optimized_arguments_file_path)
            logging_instance.info(f"Secured optimal value {best_value_for_current_parameter} for {hyperparameter_configuration.parameter_name}. Output arguments file updated.")


# -------------------------------------------------------------------------
# Configuration Section and Script Execution
# -------------------------------------------------------------------------
if __name__ == "__main__":
    
    MASTER_CONFIGURATION_SETUP = [
        HyperparameterSearchConfiguration(parameter_name="batch", is_enabled=True, values_to_test=[-1]),
        HyperparameterSearchConfiguration(parameter_name="lr0", is_enabled=True, values_to_test=[1e-4, 1e-3, 1e-2]),
        HyperparameterSearchConfiguration(parameter_name="lrf", is_enabled=True, values_to_test=[0.01, 0.05, 0.1]),
        HyperparameterSearchConfiguration(parameter_name="momentum", is_enabled=True, values_to_test=[0.8, 0.9, 0.937, 0.98]),
        HyperparameterSearchConfiguration(parameter_name="weight_decay", is_enabled=True, values_to_test=[0.0, 0.0005, 0.001]),
        
        # =====================================================================
        # PHASE 2: LOSS FUNCTION PENALTIES (Enabled for Optimization)
        # Mathematical weighting to heavily penalize rare class misclassification.
        # =====================================================================
        HyperparameterSearchConfiguration(parameter_name="cls", is_enabled=True, values_to_test=[0.5, 1.0, 1.5, 2.0]),
        HyperparameterSearchConfiguration(parameter_name="box", is_enabled=True, values_to_test=[5.0, 7.5, 10.0]),
        HyperparameterSearchConfiguration(parameter_name="dfl", is_enabled=True, values_to_test=[1.0, 1.5, 2.0]),
        HyperparameterSearchConfiguration(parameter_name="dropout", is_enabled=True, values_to_test=[0.0, 0.1, 0.2]),
        HyperparameterSearchConfiguration(parameter_name="freeze", is_enabled=True, values_to_test=[0, 1, 2]),
        HyperparameterSearchConfiguration(parameter_name="cos_lr", is_enabled=True, values_to_test=[True, False]),
        
        # =====================================================================
        # PHASE 3: SPATIAL DATA MANIPULATIONS (Enabled for Optimization)
        # Augmentations safe for satellite imagery without destroying spatial priors.
        # =====================================================================
        HyperparameterSearchConfiguration(parameter_name="copy_paste", is_enabled=True, values_to_test=[0.0, 0.3, 0.5]),
        HyperparameterSearchConfiguration(parameter_name="mosaic", is_enabled=True, values_to_test=[0.0, 0.25, 0.5]),
        HyperparameterSearchConfiguration(parameter_name="scale", is_enabled=True, values_to_test=[0.1, 0.3, 0.5]),
        HyperparameterSearchConfiguration(parameter_name="translate", is_enabled=True, values_to_test=[0.1, 0.2]),
        
        # =====================================================================
        # PHASE 4: SPECTRAL AUGMENTATIONS (Enabled for Optimization)
        # Pixel-level intensity and noise manipulation for sensor variance.
        # =====================================================================
        HyperparameterSearchConfiguration(parameter_name="hsv_h", is_enabled=True, values_to_test=[0.015, 0.05]),
        HyperparameterSearchConfiguration(parameter_name="hsv_s", is_enabled=True, values_to_test=[0.4, 0.7, 0.9]),
        HyperparameterSearchConfiguration(parameter_name="hsv_v", is_enabled=True, values_to_test=[0.4, 0.7, 0.9]),
        HyperparameterSearchConfiguration(parameter_name="erasing", is_enabled=True, values_to_test=[0.0, 0.1, 0.2]),
        HyperparameterSearchConfiguration(parameter_name="auto_augment", is_enabled=True, values_to_test=['randaugment', None]),

        # =====================================================================
        # EXCLUDED: DETRIMENTAL TO SATELLITE / RARE OBJECTS
        # These augmentations actively destroy small feature maps or labels.
        # =====================================================================
        # HyperparameterSearchConfiguration(parameter_name="multi_scale", is_enabled=False, values_to_test=[0.0, 0.5]),
        # HyperparameterSearchConfiguration(parameter_name="mixup", is_enabled=False, values_to_test=[0.0, 0.1]),
        # HyperparameterSearchConfiguration(parameter_name="cutmix", is_enabled=False, values_to_test=[0.0, 0.1]),
        # HyperparameterSearchConfiguration(parameter_name="perspective", is_enabled=False, values_to_test=[0.0, 0.001]),
        # HyperparameterSearchConfiguration(parameter_name="shear", is_enabled=False, values_to_test=[0.0, 0.2]),
        # HyperparameterSearchConfiguration(parameter_name="degrees", is_enabled=False, values_to_test=[0.0, 45.0]),
        # HyperparameterSearchConfiguration(parameter_name="bgr", is_enabled=False, values_to_test=[0.0, 0.5]),

        # =====================================================================
        # EXCLUDED: INERT MODEL HEADS
        # These do not compute gradients under task: detect.
        # =====================================================================
        # HyperparameterSearchConfiguration(parameter_name="pose", is_enabled=False, values_to_test=[12.0]),
        # HyperparameterSearchConfiguration(parameter_name="kobj", is_enabled=False, values_to_test=[1.0]),
        # HyperparameterSearchConfiguration(parameter_name="rle", is_enabled=False, values_to_test=[1.0]),
        # HyperparameterSearchConfiguration(parameter_name="angle", is_enabled=False, values_to_test=[1.0]),
        # HyperparameterSearchConfiguration(parameter_name="overlap_mask", is_enabled=False, values_to_test=),
        # HyperparameterSearchConfiguration(parameter_name="mask_ratio", is_enabled=False, values_to_test=[1]),
        # HyperparameterSearchConfiguration(parameter_name="retina_masks", is_enabled=False, values_to_test=[False]),

        # =====================================================================
        # EXCLUDED: SYSTEM, OPERATIONAL, OR STATICALLY DEFINED
        # Must be hardcoded manually before training or hold zero tuning priority.
        # =====================================================================
        # HyperparameterSearchConfiguration(parameter_name="task", is_enabled=False, values_to_test=['detect']),
        # HyperparameterSearchConfiguration(parameter_name="mode", is_enabled=False, values_to_test=['train']),
        # HyperparameterSearchConfiguration(parameter_name="model", is_enabled=False, values_to_test=['yolo26s.pt']),
        # HyperparameterSearchConfiguration(parameter_name="data", is_enabled=False, values_to_test=['xview_yolo.yaml']),
        # HyperparameterSearchConfiguration(parameter_name="epochs", is_enabled=False, values_to_test=),
        # HyperparameterSearchConfiguration(parameter_name="time", is_enabled=False, values_to_test=[None]),
        # HyperparameterSearchConfiguration(parameter_name="patience", is_enabled=False, values_to_test=),
        # HyperparameterSearchConfiguration(parameter_name="device", is_enabled=False, values_to_test=[None]),
        # HyperparameterSearchConfiguration(parameter_name="workers", is_enabled=False, values_to_test=[2]),
        # HyperparameterSearchConfiguration(parameter_name="project", is_enabled=False, values_to_test=[None]),
        # HyperparameterSearchConfiguration(parameter_name="name", is_enabled=False, values_to_test=['train']),
        # HyperparameterSearchConfiguration(parameter_name="exist_ok", is_enabled=False, values_to_test=[False]),
        # HyperparameterSearchConfiguration(parameter_name="pretrained", is_enabled=False, values_to_test=),
        # HyperparameterSearchConfiguration(parameter_name="optimizer", is_enabled=False, values_to_test=['auto']),
        # HyperparameterSearchConfiguration(parameter_name="verbose", is_enabled=False, values_to_test=),
        # HyperparameterSearchConfiguration(parameter_name="seed", is_enabled=False, values_to_test=),
        # HyperparameterSearchConfiguration(parameter_name="deterministic", is_enabled=False, values_to_test=),
        # HyperparameterSearchConfiguration(parameter_name="single_cls", is_enabled=False, values_to_test=[False]),
        # HyperparameterSearchConfiguration(parameter_name="rect", is_enabled=False, values_to_test=[False]),
        # HyperparameterSearchConfiguration(parameter_name="close_mosaic", is_enabled=False, values_to_test=[3]),
        # HyperparameterSearchConfiguration(parameter_name="resume", is_enabled=False, values_to_test=[False]),
        # HyperparameterSearchConfiguration(parameter_name="amp", is_enabled=False, values_to_test=),
        # HyperparameterSearchConfiguration(parameter_name="fraction", is_enabled=False, values_to_test=[1.0]),
        # HyperparameterSearchConfiguration(parameter_name="profile", is_enabled=False, values_to_test=[False]),
        # HyperparameterSearchConfiguration(parameter_name="compile", is_enabled=False, values_to_test=[False]),
        # HyperparameterSearchConfiguration(parameter_name="val", is_enabled=False, values_to_test=),
        # HyperparameterSearchConfiguration(parameter_name="split", is_enabled=False, values_to_test=['val']),
        # HyperparameterSearchConfiguration(parameter_name="save_json", is_enabled=False, values_to_test=[False]),
        # HyperparameterSearchConfiguration(parameter_name="conf", is_enabled=False, values_to_test=[None]),
        # HyperparameterSearchConfiguration(parameter_name="iou", is_enabled=False, values_to_test=[0.7]),
        # HyperparameterSearchConfiguration(parameter_name="max_det", is_enabled=False, values_to_test=),
        # HyperparameterSearchConfiguration(parameter_name="half", is_enabled=False, values_to_test=[False]),
        # HyperparameterSearchConfiguration(parameter_name="dnn", is_enabled=False, values_to_test=[False]),
        # HyperparameterSearchConfiguration(parameter_name="plots", is_enabled=False, values_to_test=),
        # HyperparameterSearchConfiguration(parameter_name="end2end", is_enabled=False, values_to_test=[None]),
        # HyperparameterSearchConfiguration(parameter_name="source", is_enabled=False, values_to_test=[None]),
        # HyperparameterSearchConfiguration(parameter_name="vid_stride", is_enabled=False, values_to_test=[4]),
        # HyperparameterSearchConfiguration(parameter_name="stream_buffer", is_enabled=False, values_to_test=[False]),
        # HyperparameterSearchConfiguration(parameter_name="visualize", is_enabled=False, values_to_test=[False]),
        # HyperparameterSearchConfiguration(parameter_name="augment", is_enabled=False, values_to_test=[False]),
        # HyperparameterSearchConfiguration(parameter_name="agnostic_nms", is_enabled=False, values_to_test=[False]),
        # HyperparameterSearchConfiguration(parameter_name="classes", is_enabled=False, values_to_test=[None]),
        # HyperparameterSearchConfiguration(parameter_name="embed", is_enabled=False, values_to_test=[None]),
        # HyperparameterSearchConfiguration(parameter_name="show", is_enabled=False, values_to_test=[False]),
        # HyperparameterSearchConfiguration(parameter_name="save_frames", is_enabled=False, values_to_test=[False]),
        # HyperparameterSearchConfiguration(parameter_name="save_txt", is_enabled=False, values_to_test=[False]),
        # HyperparameterSearchConfiguration(parameter_name="save_conf", is_enabled=False, values_to_test=[False]),
        # HyperparameterSearchConfiguration(parameter_name="save_crop", is_enabled=False, values_to_test=[False]),
        # HyperparameterSearchConfiguration(parameter_name="show_labels", is_enabled=False, values_to_test=),
        # HyperparameterSearchConfiguration(parameter_name="show_conf", is_enabled=False, values_to_test=),
        # HyperparameterSearchConfiguration(parameter_name="show_boxes", is_enabled=False, values_to_test=),
        # HyperparameterSearchConfiguration(parameter_name="line_width", is_enabled=False, values_to_test=[None]),
        # HyperparameterSearchConfiguration(parameter_name="format", is_enabled=False, values_to_test=['torchscript']),
        # HyperparameterSearchConfiguration(parameter_name="keras", is_enabled=False, values_to_test=[False]),
        # HyperparameterSearchConfiguration(parameter_name="optimize", is_enabled=False, values_to_test=[False]),
        # HyperparameterSearchConfiguration(parameter_name="int8", is_enabled=False, values_to_test=[False]),
        # HyperparameterSearchConfiguration(parameter_name="dynamic", is_enabled=False, values_to_test=[False]),
        # HyperparameterSearchConfiguration(parameter_name="simplify", is_enabled=False, values_to_test=),
        # HyperparameterSearchConfiguration(parameter_name="opset", is_enabled=False, values_to_test=[None]),
        # HyperparameterSearchConfiguration(parameter_name="workspace", is_enabled=False, values_to_test=[None]),
        # HyperparameterSearchConfiguration(parameter_name="nms", is_enabled=False, values_to_test=[False]),
        # HyperparameterSearchConfiguration(parameter_name="warmup_epochs", is_enabled=False, values_to_test=[3.0]),
        # HyperparameterSearchConfiguration(parameter_name="warmup_momentum", is_enabled=False, values_to_test=[0.8]),
        # HyperparameterSearchConfiguration(parameter_name="warmup_bias_lr", is_enabled=False, values_to_test=[0.1]),
        # HyperparameterSearchConfiguration(parameter_name="nbs", is_enabled=False, values_to_test=),
        # HyperparameterSearchConfiguration(parameter_name="flipud", is_enabled=False, values_to_test=[0.5]),
        # HyperparameterSearchConfiguration(parameter_name="fliplr", is_enabled=False, values_to_test=[0.5]),
        # HyperparameterSearchConfiguration(parameter_name="copy_paste_mode", is_enabled=False, values_to_test=['flip']),
        # HyperparameterSearchConfiguration(parameter_name="cfg", is_enabled=False, values_to_test=[None]),
        # HyperparameterSearchConfiguration(parameter_name="tracker", is_enabled=False, values_to_test=['botsort.yaml']),
        # HyperparameterSearchConfiguration(parameter_name="save_dir", is_enabled=False, values_to_test=['/content/dataset/xview_stratified_dataset/runs/detect/train']),
        # HyperparameterSearchConfiguration(parameter_name="save", is_enabled=False, values_to_test=),
        # HyperparameterSearchConfiguration(parameter_name="save_period", is_enabled=False, values_to_test=[-1]),
        # HyperparameterSearchConfiguration(parameter_name="cache", is_enabled=False, values_to_test=[False])
    ]

    optimizer_application = CoordinateSearchOptimizer(
        initial_arguments_file_path=Path("args.yaml"),
        optimized_arguments_file_path=Path("optimized_args.yaml"),
        hyperparameter_search_space=MASTER_CONFIGURATION_SETUP
    )

    optimizer_application.execute_optimization_sequence()