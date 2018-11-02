// Databricks notebook source
import java.time._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._

import com.gyg.discovery.funnelanalysis.DataPreProcessing._
import com.gyg.discovery.funnelanalysis.FunnelAnalysis
import com.gyg.discovery.funnelanalysis.actions._

val searchSourcePattern = "searchSource=(\\d+)"
val tourRegexPattern = "-t(\\d+)"
val locationRegexPattern = "-l(\\d+)"
val categoryRegexPattern = "-tc(\\d+)"

// COMMAND ----------

dbutils.widgets.text("startDate", "2018-10-01", "1) Start Date (>=)")
dbutils.widgets.text("endDate", "2018-10-22", "2) End Date (<)")
implicit val spk = spark

def udfToDateTZ(TZ: String = "UTC") = udf((ts: java.sql.Timestamp) => {
  val dateFormatter = java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd").withZone(java.time.ZoneId.of(TZ))
  dateFormatter.format(java.time.Instant.ofEpochMilli(ts.getTime()))
})

//val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd")
val startDate = dbutils.widgets.get("startDate")
val endDate = dbutils.widgets.get("endDate")
val pingLog = sqlContext.read.table("logs_ping_all").filter($"date" >= startDate && $"date" < endDate && $"event_name".isin("BookAction", "ActivityView"))
  .select($"event_name", $"container_name", $"id", $"action", $"header", $"date", $"user", $"tour_id", $"view_id")

val funnelRawData = pingLog.select(
  $"user.visitor_id".alias("visitor_id"),
  $"user.session_id".alias("session_id"),
  $"date",
  $"header.timestamp".alias("timestamp"),
  $"header.platform".alias("platform"),
  $"event_name",
  $"header.current_url".alias("current_url"),
  $"header.referrer".alias("referrer"),
  $"tour_id".cast(StringType),
  $"id",
  $"action",
  $"view_id".cast(StringType),
  $"container_name"
)

// COMMAND ----------

val isActivityPage = new FieldEqualsCondition("Visited ADP", "event_name", "ActivityView")
                                          
val extractVisitedTours = new ExtractionValueFromField(condition = Some(isActivityPage), "tour_id", "visited_tour_ids")
val trackADPs = new TrackingStart(extractVisitedTours)
val finishTrackingOfADPs = new TrackingEnd(extractVisitedTours, false)

val isBooking = new FieldEqualsCondition("Has Booking", "event_name", "BookAction")

// COMMAND ----------

val funnelInput = funnelRawData
  .transform(groupByAndSort(sortingColumns = Seq("timestamp"), eventGroupBy =  Seq("visitor_id"), breakGroupBy = Seq("platform")))

// COMMAND ----------

val funnelResult = new FunnelAnalysis(
      actions = Seq(trackADPs, isActivityPage, isBooking, finishTrackingOfADPs))(funnelInput)
  .withColumnRenamed("step0", "isADP")
  .withColumnRenamed("step1", "isBooking")
  .cache

// COMMAND ----------

val bookers = funnelResult.filter($"isBooking" && size($"state"("visited_tour_ids")) > 3).select($"visitor_id", $"platform", $"state"("visited_tour_ids").alias("visited_tour_ids"))
val nonBookers = funnelResult.filter(! $"isBooking" && size($"state"("visited_tour_ids")) > 3).select($"visitor_id", $"platform", $"state"("visited_tour_ids").alias("visited_tour_ids"))

// COMMAND ----------

bookers.write.mode("overwrite").parquet("/dbfs/mnt/data-shared/derived/recommendations/reco_study_bookers")
nonBookers.write.mode("overwrite").parquet("/dbfs/mnt/data-shared/derived/recommendations/reco_study_non_bookers")

// COMMAND ----------

