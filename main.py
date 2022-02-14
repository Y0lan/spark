from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import StopWordsRemover

spark = SparkSession \
    .builder \
    .appName("Projet") \
    .master("local[*]") \
    .getOrCreate()

csvFile = './full.csv'
df = spark.read.option("mode", "DROPMALFORMED").csv(csvFile, multiLine=True, header=True, escape='"')


# affiche les 10 projets Github avec le plus de commit
def show_top10_github_project_with_most_commit():
    df.groupBy("repo").count().sort(column("count").desc()).show(n=10)


print()
print()
print("#############")
print("TOP 10 GITHUB PROJECT WITH MOST COMMIT")
print("#############")
show_top10_github_project_with_most_commit()


# affiche le plus gros contributeur (+ de commit) sur le projet apache/spark
def show_biggest_apache_spark_contributor():
    df.filter(df.repo == "apache/spark") \
        .groupBy("author") \
        .count() \
        .sort(column("count").desc()) \
        .show(n=1)


print()
print()
print("#############")
print("BIGGEST CONTRIBUTOR OF APACHE/SPARK PROJECT")
print("#############")
show_biggest_apache_spark_contributor()


# affiche les plus gros contributeur (+ de commit) sur le projet apache/spark
# sur les 6 derniers mois
def show_biggest_contributors_apache_spark_last_6_months():
    spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

    # AUCUN COMMIT SUR APACHE/SPARK DANS LE CSV DANS LES 6 DERNIERS MOIS
    df.withColumn("date", to_date(df.date, 'EEE MMM dd H:mm:ss yyyy ZZZZZ')) \
        .filter(column("repo") == "apache/spark") \
        .filter(column("date") >= add_months(current_date(), -24)) \
        .groupBy("author").count().sort(column("count").desc()).show(n=20)


print()
print()
print("#############")
print("20 BIGGEST CONTRIBUTOR ON APACHE/SPARK ON THE LAST 24 MONTHS")
print("BECAUSE THERE IS NO COMMIT MADE ON APACHE/SPARK IN THE LAST 6 MONTHS")
print("#############")
show_biggest_contributors_apache_spark_last_6_months()


def most_used_word_in_commit_message(df_in_params):
    # remove all non alphabetical char
    df_ = df_in_params.withColumn("message", regexp_replace(col("message"), "[^A-Za-z]+", " "))
    # remove all \n and \r
    df_ = df_.withColumn("message", regexp_replace(col("message"), "[\n\r]", " "))
    # remove spaces from words
    df_ = df_.withColumn("message", trim(col("message")))
    # words to lowercase and split
    df_ = df_.withColumn('message', split(lower(col('message')), ' '))
    # remove all null rows
    df_ = df_.dropna()

    # remove stopwords
    remover = StopWordsRemover(inputCol="message", outputCol="tmp")
    df_ = remover.transform(df_)

    df_ = df_ \
        .select("tmp") \
        .withColumn("words", explode(col('tmp'))) \
        .filter(length(col('words')) > 2)

    df_.select("words").groupBy("words").count().sort(desc("count")).show(25)


print()
print()
print("#############")
print("25 MOST USED WORD IN COMMIT MESSAGE")
print("#############")
most_used_word_in_commit_message(df)


spark.stop()
