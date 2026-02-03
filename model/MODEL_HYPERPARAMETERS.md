# HYPERPARAMETER TUNING
Tài liệu này cung cấp thông tin chi tiết về không gian tìm kiếm và các bộ tham số tối ưu (Best Parameters) cho từng mô hình.

Chúng tôi sử dụng thư viện **Optuna** (với thuật toán TPE và cơ chế Pruning) để tự động hóa quá trình tìm kiếm.
* **Mục tiêu tối ưu (Objective):** Tối thiểu hóa *Root Mean Squared Error (RMSE)* trên tập kiểm định (Validation Set).
* **Tính tái lập (Reproducibility):** Toàn bộ quá trình thử nghiệm được cố định với random_seed=42.

## 1. Decision Tree Regressor
Quá trình dò tìm tham số cho mô hình Cây quyết định được thực hiện với **300 trials** (lần thử nghiệm). Thời gian tối ưu trung bình khoảng **4.76 phút**.

### 1.1. Direct Credits Prediction (6.37 phút)

| Tham số | Vùng tìm kiếm | Model Fresher | Model Senior |
| :--- | :--- | :--- | :--- |
| random_state | *Cố định* | 42 | 42 |
| criterion | {squared_error, friedman_mse, poisson} | friedman_mse | friedman_mse |
| splitter | {best, random} | best | best |
| max_depth | [3, 50] | 39 | 40 |
| max_leaf_nodes | [10, 500] | 180 | 380 |
| min_samples_split | [2, 40] | 2 | 30 |
| min_samples_leaf | [1, 20] | 12 | 14 |
| max_features | [0.1, 1.0] | 0.5105 | 0.6548 |
| ccp_alpha | [0.0, 0.05] | 0.0313 | 0.0111 |
| min_impurity_decrease | [0.0, 0.1] | 0.0664 | 0.0398 |
| **BEST RMSE** | - | **4.3008** | **3.8715** |

### 1.2. Gap Prediction (4.12 phút)

| Tham số | Vùng tìm kiếm | Model Fresher | Model Senior |
| :--- | :--- | :--- | :--- |
| random_state | *Cố định* | 42 | 42 |
| criterion | {squared_error, friedman_mse, poisson} | friedman_mse | friedman_mse |
| splitter | {best, random} | random | best |
| max_depth | [3, 50] | 6 | 48 |
| max_leaf_nodes | [10, 500] | 39 | 259 |
| min_samples_split | [2, 40] | 6 | 29 |
| min_samples_leaf | [1, 20] | 18 | 14 |
| max_features | [0.1, 1.0] | 0.6610 | 0.4316 |
| ccp_alpha | [0.0, 0.05] | 0.0165 | 0.0091 |
| min_impurity_decrease | [0.0, 0.1] | 0.0064 | 0.0165 |
| **BEST RMSE** | - | **4.1408** | **3.7960** |

### 1.3. Ratio Prediction (3.82 phút)

| Tham số | Vùng tìm kiếm | Model Fresher | Model Senior |
| :--- | :--- | :--- | :--- |
| random_state | *Cố định* | 42 | 42 |
| criterion | {squared_error, friedman_mse, poisson} | friedman_mse | friedman_mse |
| splitter | {best, random} | best | random |
| max_depth | [3, 50] | 45 | 32 |
| max_leaf_nodes | [10, 500] | 71 | 241 |
| min_samples_split | [2, 40] | 6 | 13 |
| min_samples_leaf | [1, 20] | 1 | 5 |
| max_features | [0.1, 1.0] | 0.8016 | 0.8173 |
| ccp_alpha | [0.0, 0.05] | 2.82e-06 | 1.12e-05 |
| min_impurity_decrease | [0.0, 0.1] | 0.0138 | 0.0332 |
| **BEST RMSE** | - | **4.0898** | **3.8017** |

## 2. Random Forest Regressor
Mô hình Rừng ngẫu nhiên được tối ưu hóa tập trung vào cấu trúc cây và phương pháp Bagging (Bootstrap Aggregating). Do độ phức tạp tính toán cao hơn, quá trình này được thực hiện với **150 trials**, tiêu tốn thời gian trung bình khoảng **359.33 phút**.

### 2.1. Direct Credits Prediction (390 phút)

| Tham số | Vùng tìm kiếm | Model Fresher | Model Senior |
| :--- | :--- | :--- | :--- |
| random_state | *Cố định* | 42 | 42 |
| n_estimators | *Cố định* | 1000 | 1000 |
| criterion | {squared_error, friedman_mse, poisson} | friedman_mse | friedman_mse |
| max_depth | [5, 50] | 13 | 13 |
| max_features | [0.1, 1.0] | 0.5210 | 0.5789 |
| min_samples_split | [2, 40] | 11 | 30 |
| min_samples_leaf | [1, 20] | 3 | 2 |
| bootstrap | {True, False} | True | True |
| ccp_alpha | [0.0, 0.05] | 0.0022 | 0.0009 |
| min_impurity_decrease | [0.0, 0.1] | 0.0696 | 0.0199 |
| max_samples | [0.5, 0.99] | 0.7620 | 0.6528 |
| **BEST RMSE** | - | **4.0586** | **3.6961** |

### 2.2. Gap Prediction (341 phút)

| Tham số | Vùng tìm kiếm | Model Fresher | Model Senior |
| :--- | :--- | :--- | :--- |
| random_state | *Cố định* | 42 | 42 |
| n_estimators | *Cố định* | 1000 | 1000 |
| criterion | {squared_error, friedman_mse, poisson} | squared_error | friedman_mse |
| max_depth | [5, 50] | 43 | 38 |
| max_features | [0.1, 1.0] | 0.5710 | 0.3401 |
| min_samples_split | [2, 40] | 33 | 37 |
| min_samples_leaf | [1, 20] | 1 | 3 |
| bootstrap | {True, False} | True | True |
| ccp_alpha | [0.0, 0.05] | 0.0095 | 0.0012 |
| min_impurity_decrease | [0.0, 0.1] | 0.0004 | 0.0813 |
| max_samples | [0.5, 0.99] | 0.9012 | 0.5005 |
| **BEST RMSE** | - | **3.9611** | **3.6829** |

### 2.3. Ratio Prediction (347 phút)

| Tham số | Vùng tìm kiếm | Model Fresher | Model Senior |
| :--- | :--- | :--- | :--- |
| random_state | *Cố định* | 42 | 42 |
| n_estimators | *Cố định* | 1000 | 1000 |
| criterion | {squared_error, friedman_mse, poisson} | poisson | friedman_mse |
| max_depth | [5, 50] | 47 | 15 |
| max_features | [0.1, 1.0] | 0.9923 | 0.4694 |
| min_samples_split | [2, 40] | 35 | 7 |
| min_samples_leaf | [1, 20] | 20 | 15 |
| bootstrap | {True, False} | True | False |
| ccp_alpha | [0.0, 0.05] | 0.0433 | 6.0e-06 |
| min_impurity_decrease | [0.0, 0.1] | 0.0522 | 0.0251 |
| max_samples | [0.5, 0.99] | 0.5002 | - |
| **BEST RMSE** | - | **5.5088** | **3.6957** |

## 3. XGBoost Regressor
Đối với XGBoost, chúng tôi thiết lập hàm mục tiêu sử dụng phân phối **Tweedie** (reg:tweedie) để phù hợp với đặc điểm phân phối của dữ liệu tín chỉ. Quá trình tối ưu tập trung sâu vào các tham số điều chuẩn (regularization) như reg_alpha, reg_lambda và tốc độ học (learning_rate).

Quá trình thực hiện dò tìm với **300 trials** trong thời gian trung bình **125.33 phút**.

### 3.1. Direct Credits Prediction (178 phút)

| Tham số | Vùng tìm kiếm | Model Fresher | Model Senior |
| :--- | :--- | :--- | :--- |
| random_state | *Cố định* | 42 | 42 |
| objective | *Cố định* | reg:tweedie | reg:tweedie |
| tree_method | *Cố định* | hist | hist |
| eval_metric | *Cố định* | rmse | rmse |
| n_estimators | *Cố định* | 4000 | 4000 |
| early_stopping_rounds | *Cố định* | 100 | 100 |
| learning_rate | [1e-4, 0.3] | 0.0248 | 0.0030 |
| tweedie_variance_power | [1.01, 1.99] | 1.2656 | 1.5685 |
| max_depth | [3, 12] | 3 | 9 |
| min_child_weight | [1, 100] | 53 | 37 |
| gamma | [1e-8, 10.0] | 8.38e-06 | 4.42e-04 |
| grow_policy | {depthwise, lossguide} | depthwise | lossguide |
| max_delta_step | [0.0, 10.0] | 0.7113 | 0.3656 |
| subsample | [0.5, 1.0] | 0.8266 | 0.8177 |
| colsample_bytree | [0.5, 1.0] | 0.7349 | 0.9143 |
| reg_alpha | [1e-8, 100.0] | 98.1272 | 6.60e-08 |
| reg_lambda | [1e-8, 100.0] | 6.49e-07 | 1.59e-05 |
| max_leaves | [16, 256] | - | 216 |
| **BEST RMSE** | - | **3.9428** | **3.6884** |

### 3.2. Gap Prediction (96 phút)

| Tham số | Vùng tìm kiếm | Model Fresher | Model Senior |
| :--- | :--- | :--- | :--- |
| random_state | *Cố định* | 42 | 42 |
| objective | *Cố định* | reg:tweedie | reg:tweedie |
| tree_method | *Cố định* | hist | hist |
| eval_metric | *Cố định* | rmse | rmse |
| n_estimators | *Cố định* | 4000 | 4000 |
| early_stopping_rounds | *Cố định* | 100 | 100 |
| learning_rate | [1e-4, 0.3] | 0.0029 | 0.0370 |
| tweedie_variance_power | [1.01, 1.99] | 1.3317 | 1.0930 |
| max_depth | [3, 12] | 12 | 8 |
| min_child_weight | [1, 100] | 25 | 7 |
| gamma | [1e-8, 10.0] | 0.0410 | 1.81e-05 |
| grow_policy | {depthwise, lossguide} | lossguide | lossguide |
| max_delta_step | [0.0, 10.0] | 0.2808 | 0.3579 |
| subsample | [0.5, 1.0] | 0.6612 | 0.7003 |
| colsample_bytree | [0.5, 1.0] | 0.9870 | 0.7676 |
| reg_alpha | [1e-8, 100.0] | 0.0003 | 0.0097 |
| reg_lambda | [1e-8, 100.0] | 0.0025 | 1.15e-05 |
| max_leaves | [16, 256] | 20 | 27 |
| **BEST RMSE** | - | **3.9297** | **3.6901** |

### 3.3. Ratio Prediction (102 phút)

| Tham số | Vùng tìm kiếm | Model Fresher | Model Senior |
| :--- | :--- | :--- | :--- |
| random_state | *Cố định* | 42 | 42 |
| objective | *Cố định* | reg:tweedie | reg:tweedie |
| tree_method | *Cố định* | hist | hist |
| eval_metric | *Cố định* | rmse | rmse |
| n_estimators | *Cố định* | 4000 | 4000 |
| early_stopping_rounds | *Cố định* | 100 | 100 |
| learning_rate | [1e-4, 0.3] | 0.0276 | 0.0256 |
| tweedie_variance_power | [1.01, 1.99] | 1.7318 | 1.6437 |
| max_depth | [3, 12] | 11 | 12 |
| min_child_weight | [1, 100] | 100 | 79 |
| gamma | [1e-8, 10.0] | 5.09e-06 | 6.69e-05 |
| grow_policy | {depthwise, lossguide} | lossguide | depthwise |
| max_delta_step | [0.0, 10.0] | 1.3878 | 0.0344 |
| subsample | [0.5, 1.0] | 0.5233 | 0.6507 |
| colsample_bytree | [0.5, 1.0] | 0.9329 | 0.5063 |
| reg_alpha | [1e-8, 100.0] | 6.02e-07 | 97.8976 |
| reg_lambda | [1e-8, 100.0] | 0.0023 | 4.7451 |
| max_leaves | [16, 256] | 236 | - |
| **BEST RMSE** | - | **3.9428** | **3.6673** |

## 4. LightGBM Regressor
Tương tự như XGBoost, LightGBM được cấu hình sử dụng thuật toán GBDT (Gradient Boosting Decision Tree) với phân phối **Tweedie**. Nhờ ưu thế về tốc độ huấn luyện, mô hình cho phép tìm kiếm không gian tham số rộng hơn trong thời gian ngắn hơn.

Quá trình tối ưu được thực hiện với **300 trials** trong thời gian trung bình **68.84 phút**.

### 4.1. Direct Credits Prediction (57.52 phút)

| Tham số | Vùng tìm kiếm | Model Fresher | Model Senior |
| :--- | :--- | :--- | :--- |
| random_state | *Cố định* | 42 | 42 |
| objective | *Cố định* | tweedie | tweedie |
| boosting_type | *Cố định* | gbdt | gbdt |
| boost_from_average | *Cố định* | True | True |
| metric | *Cố định* | rmse | rmse |
| n_estimators | *Cố định* | 4000 | 4000 |
| early_stopping_rounds | *Cố định* | 100 | 100 |
| learning_rate | [1e-4, 0.1] | 0.0602 | 0.0119 |
| tweedie_variance_power | [1.01, 1.99] | 1.2444 | 1.0863 |
| num_leaves | [10, 200] | 87 | 144 |
| max_depth | [3, 20] | 3 | 3 |
| min_child_samples | [10, 100] | 93 | 7 |
| reg_alpha | [1e-8, 10.0] | 8.32e-07 | 7.62e-05 |
| reg_lambda | [1e-8, 10.0] | 1.94e-04 | 0.0023 |
| min_split_gain | [0.0, 1.0] | 0.7551 | 0.8353 |
| subsample | [0.5, 1.0] | 0.7429 | 0.7925 |
| colsample_bytree | [0.5, 1.0] | 0.5197 | 0.7205 |
| extra_trees | {True, False} | False | True |
| max_bin | {255, 512} | 512 | 255 |
| **BEST RMSE** | - | **3.9479** | **3.6882** |

### 4.2. Gap Prediction (72 phút)

| Tham số | Vùng tìm kiếm | Model Fresher | Model Senior |
| :--- | :--- | :--- | :--- |
| random_state | *Cố định* | 42 | 42 |
| objective | *Cố định* | tweedie | tweedie |
| boosting_type | *Cố định* | gbdt | gbdt |
| boost_from_average | *Cố định* | True | True |
| metric | *Cố định* | rmse | rmse |
| n_estimators | *Cố định* | 4000 | 4000 |
| early_stopping_rounds | *Cố định* | 100 | 100 |
| learning_rate | [1e-4, 0.1] | 0.0071 | 0.0261 |
| tweedie_variance_power | [1.01, 1.99] | 1.8916 | 1.0288 |
| num_leaves | [10, 200] | 34 | 54 |
| max_depth | [3, 20] | 10 | 8 |
| min_child_samples | [10, 100] | 13 | 21 |
| reg_alpha | [1e-8, 10.0] | 6.36e-06 | 0.4863 |
| reg_lambda | [1e-8, 10.0] | 2.2203 | 1.95e-04 |
| min_split_gain | [0.0, 1.0] | 0.5563 | 0.1763 |
| subsample | [0.5, 1.0] | 0.9614 | 0.6496 |
| colsample_bytree | [0.5, 1.0] | 0.9239 | 0.9823 |
| extra_trees | {True, False} | True | True |
| max_bin | {255, 512} | 512 | 512 |
| **BEST RMSE** | - | **3.9171** | **3.6711** |

### 4.3. Ratio Prediction (77 phút)

| Tham số | Vùng tìm kiếm | Model Fresher | Model Senior |
| :--- | :--- | :--- | :--- |
| random_state | *Cố định* | 42 | 42 |
| objective | *Cố định* | tweedie | tweedie |
| boosting_type | *Cố định* | gbdt | gbdt |
| boost_from_average | *Cố định* | True | True |
| metric | *Cố định* | rmse | rmse |
| n_estimators | *Cố định* | 4000 | 4000 |
| early_stopping_rounds | *Cố định* | 100 | 100 |
| learning_rate | [1e-4, 0.1] | 0.0368 | 0.0074 |
| tweedie_variance_power | [1.01, 1.99] | 1.2428 | 1.3942 |
| num_leaves | [10, 200] | 151 | 89 |
| max_depth | [3, 20] | 8 | 7 |
| min_child_samples | [10, 100] | 54 | 16 |
| reg_alpha | [1e-8, 10.0] | 7.18e-07 | 0.2309 |
| reg_lambda | [1e-8, 10.0] | 1.0637 | 1.76e-07 |
| min_split_gain | [0.0, 1.0] | 0.0106 | 0.1842 |
| subsample | [0.5, 1.0] | 0.7396 | 0.8562 |
| colsample_bytree | [0.5, 1.0] | 0.9772 | 0.7518 |
| extra_trees | {True, False} | False | True |
| max_bin | {255, 512} | 255 | 512 |
| **BEST RMSE** | - | **3.9451** | **3.6727** |