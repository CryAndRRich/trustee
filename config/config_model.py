class CONFIG_MODEL:
    RANDOM_SEED = 42

    FIXED_PARAMS = {
        "Decision Tree": {
            "random_state": RANDOM_SEED
        },

        "Random Forest": {
            # Capacity & Core
            "n_estimators": 1000,
            
            # Execution
            "n_jobs": -1,
            "verbose": 0,
            
            # Reproducibility
            "random_state": RANDOM_SEED
        },

        "XGBoost": {
            # Core & Algorithm
            "objective": "reg:tweedie",
            "tree_method": "hist",
            
            # Metrics
            "eval_metric": "rmse",
            
            # Capacity
            "n_estimators": 4000,
            
            # Training Control & Features
            "early_stopping_rounds": 100,
            "enable_categorical": True,
            
            # Hardware
            "device": "cpu", # Hoặc DEVICE.type nếu dùng biến cho linh hoạt
            
            # Reproducibility
            "random_state": RANDOM_SEED
        },

        "LightGBM": {
            # Core & Algorithm
            "objective": "tweedie",
            "boosting_type": "gbdt",
            "boost_from_average": True,
            
            # Metrics
            "metric": "rmse",
            
            # Capacity
            "n_estimators": 4000,
            
            # Hardware & Logging
            "device": "cpu", # Hoặc DEVICE.type nếu dùng biến cho linh hoạt
            "verbosity": -1,
            
            # Reproducibility
            "random_state": RANDOM_SEED
        }
    }