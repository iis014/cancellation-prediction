# Databricks notebook source
# MAGIC %fs
# MAGIC ls FileStore/tables

# COMMAND ----------

df = spark.read.format('csv').load('/FileStore/tables//downfall_features_training_data__2_-f21e6.csv')

# COMMAND ----------

df = spark.read.format('csv') \
  .option("inferSchema", True) \
  .option("header", True) \
  .option("sep", ',') \
  .load('/FileStore/tables/downfall_features_training_data__2_-f21e6.csv')

# COMMAND ----------

temp_table_name = "downfall_features"

df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from downfall_features

# COMMAND ----------

permanent_table_name = "downfall_features_training"
df.write.format("parquet").saveAsTable(permanent_table_name)

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import col, lit
import pyspark.sql.functions as F
from pyspark.sql.window import Window

# COMMAND ----------

import mlflow

# COMMAND ----------

training = df
training = training.withColumn('label', F.when(col('shipment_status')=='Cancelled',1).otherwise(2))\
                   .withColumn('rate_per_ffe',col('penny_rate')/col('booked_ffe'))\
                   .withColumn('market_cancellation_ffes', F.sum('cancelled_ffe').over(Window.partitionBy('group_id')))\
                   .withColumn('market_cancellation_ratio',col('market_cancellation_ffes')/col('tot_volume_in_market'))\
                   .withColumn('booking_comparative_rate', F.percent_rank().over(Window.partitionBy('group_id').orderBy('rate_per_ffe'))) #recheck logic for this

# COMMAND ----------

training = training.orderBy(col('tot_volume_in_market'), ascending=1)

# COMMAND ----------

training = training.withColumn('group_num',F.dense_rank().over(Window.orderBy(col('tot_volume_in_market').desc())))

# COMMAND ----------

display(training.filter(col('group_num')==1))

# COMMAND ----------

training = training.filter((col('group_num')<=10) & (col('new_customer')=='existing customer')).withColumn('customer_canc_flag', 
                                                                                                          F.when(
 col('cust_canellation_market_ratio')>=0.50, lit('high')).when((col('cust_canellation_market_ratio')<0.50) & (col('cust_canellation_market_ratio')>=col('market_cancellation_ratio')),lit('medium')).otherwise(lit('low'))).select(
        ['cust_canellation_market_ratio', 'lead_time',
       'customer_market_share', 'departure_week', 'vp', 'group_id',
       'penny_rate', 'booked_ffe', 'WeekDay_LngDsc', 'label',
       'rate_per_ffe', 'market_cancellation_ratio','booking_comparative_rate', 'customer_canc_flag'])

# COMMAND ----------

training.write.format("parquet").saveAsTable('training_data', mode='overwrite')

# COMMAND ----------

train, test = training.randomSplit([0.80,0.20], seed=42)
train = train.withColumn('type', lit('train'))
test = test.withColumn('type', lit('test'))
df = train.union(test)

# COMMAND ----------

train_data = df.filter(col('type')=='train').drop('type')
train_data = train_data.withColumn('vp',F.when(col('vp').isNull(),lit('unknown')).otherwise(col('vp')))

# COMMAND ----------

test_data = df.filter(col('type')=='test').drop('type')
test_data = train_data.withColumn('vp',F.when(col('vp').isNull(),lit('unknown')).otherwise(col('vp')))

# COMMAND ----------

display(train_data.groupBy(col('label')).agg(F.count('group_id').alias('count')))

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, VectorIndexer, OneHotEncoderEstimator, StringIndexer
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier

# COMMAND ----------

num_features = ['lead_time','customer_market_share', 'booked_ffe','cust_canellation_market_ratio','market_cancellation_ratio','booking_comparative_rate']
cat_variables = ['vp','group_id','WeekDay_LngDsc']
label = ['label']
stages=[]
for column in train_data[cat_variables].columns:
  stringIndexer = StringIndexer(inputCol = column, outputCol = column+"_index")
  encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[column+"_vec"], dropLast=False)
  stages += [stringIndexer, encoder]

# COMMAND ----------

featureCols = [c + "_vec" for c in cat_variables] + num_features
vectorAssembler = VectorAssembler(inputCols=featureCols, outputCol="features")
#vectorIndexer = VectorIndexer(inputCol="rawFeatures", outputCol="features", maxCategories=7)

# COMMAND ----------

#rf = RandomForestRegressor(labelCol='label', featuresCol='features')
rfc = RandomForestClassifier(labelCol='label', featuresCol='features')

# COMMAND ----------

paramGrid = ParamGridBuilder()\
  .addGrid(rfc.maxDepth, [2])\
  .addGrid(rfc.numTrees, [200])\
  .build()

# COMMAND ----------

#evaluator = RegressionEvaluator(metricName="rmse", labelCol=rf.getLabelCol(), predictionCol=rf.getPredictionCol())
evaluator = MulticlassClassificationEvaluator(metricName="f1", labelCol=rfc.getLabelCol(), predictionCol=rfc.getPredictionCol())
cv = CrossValidator(estimator=rfc, evaluator=evaluator, estimatorParamMaps=paramGrid)

# COMMAND ----------

stages += [vectorAssembler, cv]
pipeline = Pipeline(stages=stages)

# COMMAND ----------

pipelinemodel = pipeline.fit(train_data)

# COMMAND ----------

va = pipelinemodel.stages[-2]
bm = pipelinemodel.stages[-1].bestModel
x = list(zip(va.getInputCols(), bm.featureImportances))
v = spark.createDataFrame([(tup[0], float(tup[1])) for tup in x])
v=v.withColumnRenamed("_1", 'feature').withColumnRenamed("_2",'importance')

# COMMAND ----------

display(v)

# COMMAND ----------

p = v.toPandas()

# COMMAND ----------

import matplotlib.pyplot as plt
x=p.feature
y=p.importance
#plt.plot(x=x, y=y)
fig=p.plot(kind='barh')
fig.xaxis
#display(fig.figure)
display(p)

# COMMAND ----------

res = pipelinemodel.transform(test_data)

# COMMAND ----------

display(res)

# COMMAND ----------

res.groupBy(col('prediction')).agg(F.count(col('label'))).show()

# COMMAND ----------

