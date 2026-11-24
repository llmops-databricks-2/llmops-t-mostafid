# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3.6: Unity Catalog Functions
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Creating Python UDFs in Unity Catalog
# MAGIC - Registering functions with `CREATE OR REPLACE FUNCTION`
# MAGIC - Calling UC functions from Spark SQL

# COMMAND ----------

from pyspark.sql import SparkSession

from arxiv_curator.config import get_env, load_config

spark = SparkSession.builder.getOrCreate()


cfg = load_config("../project_config.yml", env=get_env(spark))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

catalog = cfg.catalog
schema = cfg.schema

print(f"Using catalog: {catalog}, schema: {schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Register UC Functions
# MAGIC
# MAGIC We register four arithmetic functions as Unity Catalog Python UDFs.

# COMMAND ----------

# Add two numbers
function_name = f"{catalog}.{schema}.add_numbers"
spark.sql(f"""
    CREATE OR REPLACE FUNCTION {function_name}(a DOUBLE, b DOUBLE)
    RETURNS DOUBLE
    LANGUAGE PYTHON AS
    $$
    return a + b
    $$
    """)
print(f"Created: {function_name}")

# COMMAND ----------

# Subtract two numbers
function_name = f"{catalog}.{schema}.subtract_numbers"
spark.sql(f"""
    CREATE OR REPLACE FUNCTION {function_name}(a DOUBLE, b DOUBLE)
    RETURNS DOUBLE
    LANGUAGE PYTHON AS
    $$
    return a - b
    $$
    """)
print(f"Created: {function_name}")

# COMMAND ----------

# Multiply two numbers
function_name = f"{catalog}.{schema}.multiply_numbers"
spark.sql(f"""
    CREATE OR REPLACE FUNCTION {function_name}(a DOUBLE, b DOUBLE)
    RETURNS DOUBLE
    LANGUAGE PYTHON AS
    $$
    return a * b
    $$
    """)
print(f"Created: {function_name}")

# COMMAND ----------

# Divide two numbers (returns None on division by zero)
function_name = f"{catalog}.{schema}.divide_numbers"
spark.sql(f"""
    CREATE OR REPLACE FUNCTION {function_name}(a DOUBLE, b DOUBLE)
    RETURNS DOUBLE
    LANGUAGE PYTHON AS
    $$
    if b == 0:
        return None
    return a / b
    $$
    """)
print(f"Created: {function_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test the Functions

# COMMAND ----------

result = spark.sql(f"""
    SELECT
        {catalog}.{schema}.add_numbers(10, 3)      AS add_result,
        {catalog}.{schema}.subtract_numbers(10, 3) AS subtract_result,
        {catalog}.{schema}.multiply_numbers(10, 3) AS multiply_result,
        {catalog}.{schema}.divide_numbers(10, 3)   AS divide_result
""")
result.show()
