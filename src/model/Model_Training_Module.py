# Cơ bản & xử lý dữ liệu
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats.mstats import winsorize

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn: tiền xử lý & chia tập
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.feature_selection import SelectFromModel

# Scikit-learn: đánh giá mô hình
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

# Scikit-learn: mô hình ML
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# Mô hình nâng cao
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Bỏ cảnh báo
import warnings


# Giả định data_lr là DataFrame đã được encoded và không có NaN
def Splitting_Train_Dev_Test (X, y):
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=32)
    X_train, X_dev, y_train, y_dev = train_test_split(X_temp, y_temp, test_size=0.25, random_state=32)
    
    return X_train, y_train, X_dev, y_dev, X_test, y_test

def Standard_X (X_train, X_dev, X_test):
    X_scaler = StandardScaler ()
    X_train_scaled = X_scaler.fit_transform (X_train)
    X_dev_scaled = X_scaler.transform (X_dev)
    X_test_scaled = X_scaler.transform (X_test)
    
    return X_scaler, X_train_scaled, X_dev_scaled, X_test_scaled


def remove_outliers_iqr(df):
    """
    Loại bỏ outliers trong cột target_col của DataFrame dựa vào phương pháp IQR.
    
    Tham số:
        df (pd.DataFrame): DataFrame gốc gồm cả đặc trưng và biến mục tiêu.
        target_col (str): Tên cột biến mục tiêu.
    
    Trả về:
        pd.DataFrame: DataFrame đã loại bỏ các outliers.
    """
    Q1 = df['Market_Value'].quantile(0.25)
    Q3 = df['Market_Value'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df['Market_Value'] >= lower_bound) & (df['Market_Value'] <= upper_bound)]
    return filtered_df


def winsorize_data(y, lower_percentile=5, upper_percentile=95):
    lower_bound = np.percentile(y, lower_percentile)
    upper_bound = np.percentile(y, upper_percentile)
    y_winsorized = np.clip(y, lower_bound, upper_bound)
    return y_winsorized


def Standard_y(y_train, y_dev, y_test):
    '''
    Chuẩn hoá Standardize
    '''
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(np.array(y_train).reshape(-1, 1)).ravel()
    y_dev_scaled = y_scaler.transform(np.array(y_dev).reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(np.array(y_test).reshape(-1, 1)).ravel()

    return y_scaler, y_train_scaled, y_dev_scaled, y_test_scaled


def Log_Standard_y(y_train, y_dev, y_test):
    '''Hàm chuẩn hoá y
    - Chuẩn hoá lop1p
    - Chuẩn hoá tiếp Standardize
    '''
    y_scaler = StandardScaler()
    y_train_log = np.log1p (y_train)
    y_dev_log = np.log1p (y_dev)
    y_test_log = np.log1p (y_test)
    
    y_train_scaled = y_scaler.fit_transform(np.array(y_train_log).reshape(-1, 1)).ravel()
    y_dev_scaled = y_scaler.transform(np.array(y_dev_log).reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(np.array(y_test_log).reshape(-1, 1)).ravel()

    return y_scaler, y_train_scaled, y_dev_scaled, y_test_scaled


def inverse_log_standard_y (y_scaler, y_train_pred, y_dev_pred, y_test_pred):
    '''
    Hàm chuyển đổi chuẩn hoá
    - Inverse Standardize
    - Inverse Log1p
    '''
    y_train_pred_log = y_scaler.inverse_transform(y_train_pred.reshape(-1, 1))
    y_train_pred = np.expm1(y_train_pred_log)
    y_dev_pred_log = y_scaler.inverse_transform(y_dev_pred.reshape(-1, 1))
    y_dev_pred = np.expm1(y_dev_pred_log)
    y_test_pred_log = y_scaler.inverse_transform(y_test_pred.reshape(-1, 1))
    y_test_pred = np.expm1(y_test_pred_log)
    
    return y_train_pred, y_dev_pred, y_test_pred


def inverse_y (y_scaler, y_train_pred, y_dev_pred, y_test_pred):
    '''
    Hàm chuyển đổi chuẩn hoá
    - Inverse Standardize
    '''
    y_train_pred = y_scaler.inverse_transform(y_train_pred.reshape(-1, 1))
    y_dev_pred = y_scaler.inverse_transform(y_dev_pred.reshape(-1, 1))
    y_test_pred = y_scaler.inverse_transform(y_test_pred.reshape(-1, 1))
    
    return y_train_pred, y_dev_pred, y_test_pred


class Evaluate:
    """
    Class chứa các phương thức đánh giá mô hình hồi quy:
    - MSE: Mean Squared Error
    - RMSE: Root Mean Squared Error
    - MAE: Mean Absolute Error
    - MAPE: Mean Absolute Percentage Error
    - R2 Score: Hệ số xác định (R^2)
    """

    @staticmethod
    def mse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def mae(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def mape(y_true, y_pred):
        return mean_absolute_percentage_error(y_true, y_pred) * 100  # Kết quả dạng %

    @staticmethod
    def r2(y_true, y_pred):
        return r2_score(y_true, y_pred)

    @staticmethod
    def evaluate_all(y_train, y_train_pred, y_dev, y_dev_pred, y_test, y_test_pred):
        """
        Trả về toàn bộ metric trên 3 tập: train, dev, test
        """
        results = {
            "MSE": {
                "Train": Evaluate.mse(y_train, y_train_pred),
                "Dev": Evaluate.mse(y_dev, y_dev_pred),
                "Test": Evaluate.mse(y_test, y_test_pred),
            },
            "RMSE": {
                "Train": Evaluate.rmse(y_train, y_train_pred),
                "Dev": Evaluate.rmse(y_dev, y_dev_pred),
                "Test": Evaluate.rmse(y_test, y_test_pred),
            },
            "MAE": {
                "Train": Evaluate.mae(y_train, y_train_pred),
                "Dev": Evaluate.mae(y_dev, y_dev_pred),
                "Test": Evaluate.mae(y_test, y_test_pred),
            },
            "MAPE (%)": {
                "Train": Evaluate.mape(y_train, y_train_pred),
                "Dev": Evaluate.mape(y_dev, y_dev_pred),
                "Test": Evaluate.mape(y_test, y_test_pred),
            },
            "R2 Score": {
                "Train": Evaluate.r2(y_train, y_train_pred),
                "Dev": Evaluate.r2(y_dev, y_dev_pred),
                "Test": Evaluate.r2(y_test, y_test_pred),
            }
        }
        return results


def Print (results, model_name, best_model, data):
    print(f"Kết quả đánh giá mô hình {model_name} với hyperparmeter tối ưu:")

    # In tiêu đề
    print(f"{'Set':<10}{'MSE':>10}{'RMSE':>10}{'R2':>10}{'MAPE (%)':>12}{'MAE':>10}")
    print("-" * 62)

    # In từng dòng kết quả theo tập Train, Dev, Test
    for subset in ["Train", "Dev", "Test"]:
        mse = results["MSE"][subset]
        rmse = results["RMSE"][subset]
        r2 = results["R2 Score"][subset]
        mape = results["MAPE (%)"][subset]
        mae = results["MAE"][subset]

        print(f"{subset:<10}{mse:>10.4f}{rmse:>10.4f}{r2:>10.4f}{mape:>12.2f}{mae:>10.4f}")

def Visualize (best_model, model_name, X_train_scaled, y_train_scaled, y_train, y_dev, y_test, y_train_pred, y_dev_pred, y_test_pred):
    '''
    Lưu ý: X_train và y_train truyền vào hàm cũng phải được scaled nếu mô hình có scaled
    '''    
        # a. Dự đoán so với giá trị thực
    fig, ax = plt.subplots (1,2 , figsize=(18,8))
    ax[0].scatter(y_train, y_train_pred, color='blue', label='Train', alpha=0.5)
    ax[0].scatter(y_dev, y_dev_pred, color='green', label='Dev', alpha=0.5)
    ax[0].scatter(y_test, y_test_pred, color='red', label='Test', alpha=0.5)
    ax[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='black', linestyle='--', label='Đường lý tưởng')
    ax[0].set_xlabel('Giá trị thực (Market_Value, đơn vị gốc)')
    ax[0].set_ylabel('Giá trị dự đoán (đơn vị gốc)')
    ax[0].set_title(f'Dự đoán so với Giá trị thực {model_name}')
    ax[0].legend()
    ax[0].grid(True)

        # b. Vẽ Learning Curve
    train_sizes, train_scores, valid_scores = learning_curve(
        estimator=best_model,
        X=X_train_scaled,
        y=y_train_scaled,
        cv=5,
        scoring='r2',
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1
    )

    # Tính trung bình và độ lệch chuẩn
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std  = np.std(valid_scores, axis=1)

    ax[1].plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training score')
    ax[1].fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.2, color='blue')

    ax[1].plot(train_sizes, valid_scores_mean, 'o-', color='orange', label='Cross-validation score')
    ax[1].fill_between(train_sizes, valid_scores_mean - valid_scores_std,
                    valid_scores_mean + valid_scores_std, alpha=0.2, color='orange')

    ax[1].set_title(f'Learning Curve ({model_name})')
    ax[1].set_xlabel('Training Set Size')
    ax[1].set_ylabel('R² Score')
    ax[1].grid(True)
    ax[1].legend(loc='best')

    plt.suptitle (f'Trực quan hoá mô hình {model_name}.')
    plt.tight_layout()
    plt.show()