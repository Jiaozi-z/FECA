from recbole.quick_start import load_data_and_model
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_model_path(sample):
    if isinstance(sample, np.ndarray):
        feature_key = tuple(sample.astype(int))
    else:
        feature_key = sample 
    

    
    model_paths = {
        # 更新模型路径映射，所有模型都关闭Price和Time控制
        (1, 0, 0, 1, 1): 'saved/FECA-full.pth',              # 完整版本
        (0, 0, 0, 1, 1): 'saved/FECA-no-query.pth',         # 无query
        (1, 0, 0, 0, 1): 'saved/FECA-no-exposure.pth',      # 无exposure
        (0, 0, 0, 0, 1): 'saved/FECA-no-query-exposure.pth',# 无query和exposure
        (1, 0, 0, 1, 0): 'saved/FECA-no-profile.pth',       # 无profile
        (0, 0, 0, 1, 0): 'saved/FECA-no-query-profile.pth', # 无query和profile
        (1, 0, 0, 0, 0): 'saved/FECA-no-exposure-profile.pth', # 无exposure和profile
        (0, 0, 0, 0, 0): 'saved/FECA-base.pth',             # 基础版本
    }
    
    logger.info(f"Original key: {feature_key}, Normalized key: {feature_key}")
    return model_paths.get(feature_key, 'saved/FECA-base.pth')

def load_all_models():
    """预先加载所有模型"""
    logger.info("Loading all models...")
    models = {}
    
    # 需要加载的模型配置
    model_configs = [
        (1, 0, 0, 1, 1),  # 完整版本
        (0, 0, 0, 1, 1),  # 无query
        (1, 0, 0, 0, 1),  # 无exposure
        (0, 0, 0, 0, 1),  # 无query和exposure
        (1, 0, 0, 1, 0),  # 无profile
        (0, 0, 0, 1, 0),  # 无query和profile
        (1, 0, 0, 0, 0),  # 无exposure和profile
        (0, 0, 0, 0, 0),  # 基础版本
    ]
    
    for config in model_configs:
        model_path = get_model_path(config)
        logger.info(f"Attempting to load model from path: {model_path}")
        try:
            config_dict, model, dataset, _, _, test_data = load_data_and_model(model_path)
            models[config] = model
            logger.info(f"Successfully loaded model: {model_path}")
        except Exception as e:
            logger.error(f"Error loading model {model_path}: {str(e)}")
            continue
    
    return models, test_data

def score_func(X):
    if not hasattr(score_func, 'models'):
        logger.info("Loading models and data...")
        score_func.models, score_func.test_data = load_all_models()
    
    scores = []
    
    PRICE_CONTROL_BASELINE = 80
    TIME_CONTROL_BASELINE = 75
    PRICE_CONTROL_WEIGHT = 0.007084307726472616
    TIME_CONTROL_WEIGHT = 0.019106673076748848
    
    for sample_idx, sample in enumerate(X):
        logger.info(f"Processing sample {sample_idx} with features {sample}")
        try:
            # 生成特征键
            key = tuple(sample.astype(int))
            logger.info(f"Generated key for model lookup: {key}")
            
            # 映射特征键，忽略Price和Time
            normalized_key = (
                key[0],  # Query
                0,       # Price 统一设为0
                0,       # Time 统一设为0
                key[3],  # exposure
                key[4]   # Profile
            )
            logger.info(f"Normalized key for model lookup: {normalized_key}")
            
            # 获取模型
            model = score_func.models.get(normalized_key, score_func.models.get((0, 0, 0, 0, 0)))
            if model is None:
                logger.warning(f"No model found for normalized key: {normalized_key}, using base model")
            
            with torch.no_grad():
                user_interaction = score_func.test_data.dataset[0:1]
                logger.info(f"User interaction data: {user_interaction}")
                scores_tensor = model.full_sort_predict(user_interaction)
                
                if scores_tensor is not None:
                    scores_tensor = scores_tensor.clone()
                    item_prices = model.item_price.unsqueeze(0)
                    item_times = model.item_time.unsqueeze(0)
                    
                    # 根据Price和Time特征调整分数
                    if sample[1] == 1:
                        price_diff = (item_prices - PRICE_CONTROL_BASELINE) / PRICE_CONTROL_BASELINE
                        price_effect = torch.where(price_diff > 0, price_diff, torch.zeros_like(price_diff)) * PRICE_CONTROL_WEIGHT
                        scores_tensor -= price_effect.view(-1)
                    
                    if sample[2] == 1:
                        time_diff = (item_times - TIME_CONTROL_BASELINE) / TIME_CONTROL_BASELINE
                        time_effect = torch.where(time_diff > 0, time_diff, torch.zeros_like(time_diff)) * TIME_CONTROL_WEIGHT
                        scores_tensor -= time_effect.view(-1)
                    
                    score = scores_tensor.mean().item()
                    logger.info(f"Score for sample {sample_idx}: {score}")
                    scores.append(score)
                else:
                    logger.warning(f"Model returned None for sample {sample_idx}")
                    scores.append(0.0)
                    
        except Exception as e:
            logger.warning(f"Error processing sample {sample_idx} with features {sample}: {str(e)}")
            scores.append(0.0)
    
    result = np.array(scores)
    logger.info(f"Final scores shape: {result.shape}, values: {result}")
    return result

def main():
    # 设置随机种子
    np.random.seed(2024)
    torch.manual_seed(2024)
    
    # 定义背景数据
    background_data = np.array([
        [1, 1, 1, 1, 1],  # 全版本：所有特征都开启
        [1, 1, 0, 1, 1],  # 去掉time
        [1, 0, 0, 1, 1],  # 去掉time和price
        [1, 0, 0, 0, 1],  # 去掉time、price和exposure
        [1, 0, 0, 0, 0],  # 去掉time、price、exposure和profile
        [0, 0, 0, 0, 0],  # 只保留基础User-Item交互
    ])
    
    try:
        # 创建SHAP解释器
        logger.info("Creating SHAP explainer...")
        explainer = shap.KernelExplainer(
            score_func, 
            background_data,
            nsamples=200  # 增加采样次数
        )
        
        # 选择要解释的样本
        test_samples = np.array([
            [1, 1, 1, 1, 1],  # 完整模型
        ])
        
        # 计算SHAP值
        logger.info("Calculating SHAP values...")
        shap_values = explainer.shap_values(
            test_samples,
            nsamples=200,  # 增加采样次数
            l1_reg="num_features(5)"  # 添加正则化
        )
        
        # 确保shap_values是一个numpy数组
        if not isinstance(shap_values, np.ndarray):
            shap_values = np.array(shap_values)
        
        # 如果shap_values是一维数组，将其reshape为2维
        if len(shap_values.shape) == 1:
            shap_values = shap_values.reshape(1, -1)
            
        
        # 保存结果
        save_dir = 'shap_results'
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存SHAP值
        np.save(os.path.join(save_dir, 'shap_values.npy'), shap_values)
        
        # 创建特征名称列表
        feature_names = ['Query', 'Price', 'Time', 'Exposure', 'Profile']
        
        # 绘制SHAP柱状图
        # 计算每个特征的平均 SHAP 值
        average_shap_values =np.abs( np.mean(shap_values, axis=0))
        
        # 绘制SHAP柱状图
        logger.info("创建SHAP柱状图...")
        shap.bar_plot(
            average_shap_values, 
            feature_names=feature_names,
            show=False
        )
        plt.xlabel("Average SHAP Value")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'shap_bar_plot.pdf'))
        plt.close()
        
        # 打印平均SHAP值
        print("\nAverage SHAP Values:")
        for i, feature in enumerate(feature_names):
            print(f"{feature}: {average_shap_values[i]}")
        
        # 计算并打印特征重要性（绝对SHAP值的平均值）
        feature_importance = np.abs(shap_values).mean(0)
        print("\nFeature Importance:")
        for i, feature in enumerate(feature_names):
            print(f"{feature}: {feature_importance[i]}")
            
        # # 保存特征重要性到文件
        # importance_dict = {feature: float(importance) 
        #                  for feature, importance in zip(feature_names, feature_importance)}
        # np.save(os.path.join(save_dir, 'feature_importance.npy'), importance_dict)
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()