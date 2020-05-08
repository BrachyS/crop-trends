# crop-trends
Time series and network analysis on worldwide trade data of agricultural commodities (In progress).

**Problem statement**
Food security is becoming a pressing issue especially in some developing countries in tropical areas, due to climate change and a fast-growing population that may affect both crop production and demand. Therefore, it is important to understand how food availability may change through domestic production and international trade under future climate scenarios. 

In this project, we will examine the time series data of crop production and yield among countries and predict future trends. Current and future trade network among countries will also be analyzed and modelled. Lastly, climate change scenarios will be incorporated to improve accuracy of the forecasts. 

**Potential Clients**
Government officials and policy makers, NGOs and research institutions interested in strategies to improve the economic outlook of countries whose agricultural production and food security may be threatened by future climate conditions. 

Dataset used in this project: http://www.fao.org/faostat/en/#data/TM

Access to the project's dashboard here: [Global Agricultural Commodities Export Forecast](http://18.144.177.199:8501/), where you can select a country, an export/import item and metrics, and optimize an ARIMA model to predict trends into year 2025 (check back for updates!).

Outline of codes:
[1. Data Cleaning](https://github.com/BrachyS/crop-trends/blob/master/notebooks/1_data-cleaning-agtrend.ipynb)
[2. Exploratory analysis](https://github.com/BrachyS/crop-trends/blob/master/notebooks/2_exploratory-analysis-agtrade.ipynb)
[3. Arima modeling for five top crops](https://github.com/BrachyS/crop-trends/blob/master/notebooks/3_top_crops_arima.ipynb)
[4. Streamline ARIMA modeling for all item-by-country combinations](https://github.com/BrachyS/crop-trends/blob/master/notebooks/4._streamline_ARIMA_modeling.ipynb)
[5. Next steps](In progress)
[Dashboard](https://github.com/BrachyS/crop-trends/blob/master/dashboard/dash_export.py)