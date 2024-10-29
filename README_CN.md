# To bear or to share? A Fair, Efficient, and ConsiderateApproach for Personalized Patient-PhysicianRecommendations.

## 简介

学者们普遍认识到推荐系统会放大马太效应,小众选项失去可见度,热门内容获得更多关注,用户偏好被动强化。然而,在医疗服务提供者推荐中,这些偏见对所有人都有害:过度曝光的医生面临倦怠,患者忍受漫长等待时间,许多合格的医生未被充分利用,这代表了一个未被开发的多数群体。为解决这个问题,我们提出了FECA,一种公平、高效且体贴的个性化医患推荐方法。

## 依赖项

本项目依赖以下主要Python包:

- `aiohttp>=3.10.5`
- `numpy>=1.24.3`
- `pandas>=2.2.2`
- `scikit-learn>=1.5.1`
- `tensorflow>=2.17.0`
- `torch>=1.12.1`
- `transformers>=4.44.2`
- `matplotlib>=3.9.2`
- `seaborn>=0.13.2`
- `requests>=2.32.3`

请确保在运行项目前安装这些依赖项。

## 使用方法

本文相关模型是基于中国人民大学recbole代码库的基础代码设计和训练的。本文提出的模型称为FECA。FECA和其他基线模型代码可以在`recbole/model/gemeral_recommender`文件夹中找到。相关训练设置在`config/model_config.yaml`和`recbole/properties/model/FECA.yaml`中。运行`run_customize_model.py`开始训练和测试。

## 引用
本项目使用了[RecBole](https://github.com/RUCAIBox/RecBole)代码库的代码。RecBole是一个统一的、全面的、高效的推荐系统库,支持各种推荐算法和数据集。
特别感谢RecBole团队的贡献。