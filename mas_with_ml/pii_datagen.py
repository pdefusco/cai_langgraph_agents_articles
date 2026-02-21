#****************************************************************************
# (C) Cloudera, Inc. 2020-2023
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# #  Author(s): Paul de Fusco
#***************************************************************************/

import os
import numpy as np
import pandas as pd
from datetime import datetime
from pyspark.sql.types import LongType, IntegerType, StringType
from pyspark.sql import SparkSession
import dbldatagen as dg
import dbldatagen.distributions as dist
from dbldatagen import FakerTextFactory, DataGenerator, fakerText
from faker.providers import bank, credit_card, currency
import cml.data_v1 as cmldata


class BankDataGen:

    '''Class to Generate Banking Data'''

    def __init__(self, username, dbname, connectionName):
        self.username = username
        self.dbname = dbname
        self.connectionName = connectionName


    def dataGen(self, spark, shuffle_partitions_requested=2, partitions_requested=2, data_rows=10000):

        FakerTextUS = FakerTextFactory(
            locale=['en_US'],
            providers=[bank, credit_card, currency]
        )

        spark.conf.set("spark.sql.shuffle.partitions", shuffle_partitions_requested)

        fakerDataspec = (
            DataGenerator(spark, rows=data_rows, partitions=partitions_requested)
                .withColumn("customer_id", "long", minValue=1, step=1)
                .withColumn("first_name", "string", text=fakerText("first_name"))
                .withColumn("last_name", "string", text=fakerText("last_name"))
                .withColumn("full_name", "string", text=fakerText("name"))
                .withColumn("email", "string", text=fakerText("email"))
                .withColumn("phone_number", "string", text=fakerText("phone_number"))
                .withColumn("ssn", "string", text=fakerText("ssn"))
                .withColumn("street_address", "string", text=fakerText("street_address"))
                .withColumn("city", "string", text=fakerText("city"))
                .withColumn("state", "string", text=fakerText("state_abbr"))
                .withColumn("zip_code", "string", text=fakerText("postcode"))
                .withColumn("country", "string", text=fakerText("country"))
                .withColumn("company", "string", text=fakerText("company"))
                .withColumn("job_title", "string", text=fakerText("job"))
                .withColumn("risk_tier", "string",
                    values=["LOW", "MEDIUM", "HIGH"],
                    weights=[70, 20, 10],
                    random=True)
                    )

        df = fakerDataspec.build()

        return df


    def createSparkConnection(self):
        """
        Method to create a Spark Connection using CML Data Connections
        """

        from pyspark import SparkContext
        SparkContext.setSystemProperty('spark.executor.cores', '2')
        SparkContext.setSystemProperty('spark.executor.memory', '4g')

        import cml.data_v1 as cmldata
        conn = cmldata.get_connection(self.connectionName)
        spark = conn.get_spark_session()

        return spark


    def saveFileToCloud(self, df):
        """
        Method to save credit card transactions df as csv in cloud storage
        """
        #df.write.format("csv").mode('overwrite').save(self.storage + "/bank_fraud_demo/" + self.username)
        pass


    def createDatabase(self, spark):
        """
        Method to create database before data generated is saved to new database and table
        """

        spark.sql("CREATE DATABASE IF NOT EXISTS {}".format(self.dbname))

        print("SHOW DATABASES LIKE '{}'".format(self.dbname))
        spark.sql("SHOW DATABASES LIKE '{}'".format(self.dbname)).show()


    def createOrReplace(self, df):
        """
        Method to create or append data to the BANKING TRANSACTIONS table
        The table is used to simulate batches of new data
        The table is meant to be updated periodically as part of a CML Job
        """

        try:
            df.write \
                .format("parquet") \
                .saveAsTable("{0}.customers".format(self.dbname))

        except Exception:
            taskMetricsDf.write \
                .mode("append") \
                .format("parquet") \
                .saveAsTable("{0}.customers".format(self.dbname))


    def validateTable(self, spark):
        """
        Method to validate creation of table
        """
        print("SHOW TABLES FROM '{}'".format(self.dbname))
        spark.sql("SHOW TABLES FROM {}".format(self.dbname)).show()


def main():

    USERNAME = os.environ["PROJECT_OWNER"]
    DBNAME = "PII_"+USERNAME
    CONNECTION_NAME = "pdf-feb-aw-dl"

    # Instantiate BankDataGen class
    dg = BankDataGen(USERNAME, DBNAME, CONNECTION_NAME)

    # Create CML Spark Connection
    spark = dg.createSparkConnection()

    # Create Banking Transactions DF
    df = dg.dataGen(spark)

    # Create Spark Database
    dg.createDatabase(spark)

    # Create Iceberg Table in Database
    dg.createOrReplace(df)

    # Validate Iceberg Table in Database
    dg.validateTable(spark)


if __name__ == '__main__':
    main()
