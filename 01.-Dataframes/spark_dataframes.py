# Databricks notebook source
#importar clases para tipos y funciones
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType
import pyspark.sql.functions as F
import uuid
from pyspark.sql.functions import udf

# COMMAND ----------

# MAGIC %md
# MAGIC Las funciones definidas por el usuario (UDF) permiten reutilizar y compartir código que amplía la funcionalidad integrada en Databricks. Use UDF para realizar tareas específicas, como cálculos complejos, transformaciones o manipulaciones de datos personalizadas. 
# MAGIC Usamos UDF para la lógica que es difícil de expresar con funciones integradas de Apache Spark. Las funciones integradas de Apache Spark están optimizadas para el procesamiento distribuido y ofrecen un mejor rendimiento a escala

# COMMAND ----------

def get_uuid():
    return str(uuid.uuid4())
uuid_udf = udf(get_uuid, StringType())

# COMMAND ----------

#ejemplo de creación de un dataframe
#definimos su schema
schema=StructType(
    [
  StructField("id",StringType(),True),
  StructField("alumno",StringType(),True),
  StructField("materia",StringType(),True),
  StructField("calificacion",DoubleType(),True),
  StructField("asistencias_mes",ArrayType(
    StructType([
            StructField("mes", StringType(), True),
            StructField("asistencias", IntegerType(), True)
        ])
  ),True),
    ]   
  )
#creamos la data en una lista de listas
data=[
  ["408082267","Juan Pérez","Matemáticas",8.0,[('Enero',19),('Febrero',18)]],
  ["2","Roberto","Español",8.6,[('Enero',15),('Febrero',20)]],
  ["3","Juan Pérez", "Español",9.0,[('Enero',19),('Febrero',18)]],
  ["4","Jose Luis","Español",8.0,[('Enero',17),('Febrero',19)]],
  ["5","Jaime","",None,[('Enero',16),('Febrero',15)]],
  ]
#creamos el dataframe
df_alumnos=spark.createDataFrame(data,schema)
"""
Un identificador único universal o universally unique identifier (UUID) es un número de 128 bits, con lo que el número de posibles UUID diferentes sería de 2128
"""
df_alumnos=df_alumnos.withColumn("id_uuid",uuid_udf())
df_alumnos.display()

# COMMAND ----------


#creación de un segundo dataframe
schema=StructType(
    [
  StructField("id",StringType(),True),
  StructField("alumno",StringType(),True),
  StructField("edad",DoubleType(),True)
    ]   
  )

data=[["1","Juan Pérez",18.0],["2","Roberto",19.0],["3","Juan Pérez",21.0],["4","Jose Luis",22.0]]
df_edad=spark.createDataFrame(data,schema)
df_edad.display()

# COMMAND ----------

# Con withColumn se agregan columnas y con un valor constante usando lit, ambas columnas agregaron valor de tiempo.
df_alumnos=(df_alumnos.withColumn("timestamp",F.lit(F.current_timestamp()))
    .withColumn("unix_timestamp",F.lit(F.unix_timestamp() ))
)
df_alumnos.limit(5).display()

# COMMAND ----------

"""se agregan tres columnas, una de fecha, otra es un timestamp del uso horario de México
y finalmente la conversión del unix_timestamp a formato de timestamp"""

df_alumnos=(df_alumnos.withColumn("date",F.to_date(F.col("timestamp"),format='yyyy-MM-dd'))
 .withColumn("timestamp_mexico",F.from_utc_timestamp(F.col("timestamp"), "America/Mexico_City"))
 .withColumn("timestamp_from_unix",F.from_unixtime(F.col("unix_timestamp")))
)
df_alumnos.display()

# COMMAND ----------

#filtrado de información de un conjunto de datos
df_alumnos.filter(F.col("calificacion").alias("calif")>=9).display()
df_alumnos.filter("calificacion>=9").display() 
df_alumnos.filter((df_alumnos["calificacion"]>=8) | (df_alumnos["id"]>='2') ).display() 

# COMMAND ----------

df_alumnos.filter(F.col("calificacion").isNotNull()).display()

# COMMAND ----------


df_alumnos.where(F.col("materia").isNotNull()).display()

# COMMAND ----------

#Explode sobre columnas que son arrays para obetener un registro de asistencia por fila
asitencia_mes=df_alumnos.select("id",F.explode(F.col("asistencias_mes")).alias("asistencias"))
asitencia_mes.limit(10).display()
df_final=asitencia_mes.select(F.col("asistencias.mes").alias("mes"),F.col("asistencias.asistencias"))
df_final.display()

# COMMAND ----------

#estructuracion de la información
asitencia_mes=(asitencia_mes.withColumn("mes",F.col("asistencias.mes"))
               .withColumn("asistencia",F.col("asistencias.asistencias"))
).drop(*["asistencias"])
asitencia_mes.display()

# COMMAND ----------

df_alumnos.orderBy("calificacion",ascending=False).display()

# COMMAND ----------

#agregaciones
df_alumnos.groupBy("alumno").count().display()
df_alumnos.groupBy("materia").mean("calificacion").display()

# COMMAND ----------

#agregaciones
df_alumnos.groupBy("materia").mean().select(F.col("materia"),F.col("avg(calificacion)").alias("media")).display()
df_alumnos.groupBy("materia").agg(F.count("materia")).display()
df_alumnos.groupBy("materia").agg(F.mean("calificacion")).display()
df_alumnos.groupBy("materia").agg(F.max("calificacion").alias("maximo")).display()

# COMMAND ----------

df_alumnos.createOrReplaceTempView("alumnos")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from alumnos

# COMMAND ----------

#creacion de un directorio
dbutils.fs.mkdirs('dbfs:/FileStore/tables/test')

# COMMAND ----------

# MAGIC %md
# MAGIC Tipos de Joins
# MAGIC * INNER JOIN : Devuelve los registros que tienen valores coincidentes en ambas tablas.
# MAGIC * LEFT JOIN : Devuelve todos los registros de la tabla izquierda y los registros coincidentes de la tabla derecha.
# MAGIC * RIGHT JOIN: Devuelve todos los registros de la tabla derecha y los registros coincidentes de la tabla izquierda.
# MAGIC * FULL OUTER: Devuelve todos los registros cuando hay una coincidencia en cualquiera de las tablas, izquierda o derecha.

# COMMAND ----------

df_alumnos=df_alumnos.alias("df_alumnos")
df_edad=df_edad.alias("df_edad")
df_inner_join=df_alumnos.join(df_edad, df_alumnos.id == df_edad.id, "inner") 
df_inner_join.select("df_alumnos.id").display()

# COMMAND ----------

df_alumnos.join(df_edad, df_alumnos.id == df_edad.id, "left").display()

# COMMAND ----------

df_alumnos.join(df_edad, df_alumnos.id == df_edad.id, "right").display() 

# COMMAND ----------

df_alumnos.join(df_edad, df_alumnos.id == df_edad.id, "fullouter").display() 
