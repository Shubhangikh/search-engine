/** Build a search engine for the movie plot summaries available from the 
  * Carnegie Movie Summary Corpus [[http://www.cs.cmu.edu/~ark/personas/]]
  * by computing tf-idf using MapReduce
  */
  
import org.apache.spark.sql.functions.split
import org.apache.spark.sql.functions.regexp_replace
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.lower
import org.apache.spark.sql.functions.{col}
import org.apache.spark.ml.feature.StopWordsRemover
import sqlContext.implicits._
import scala.math

// Read files
val movieData = spark.read.option("header","true").option("inferSchema","true").option("delimiter","\t").csv("/FileStore/tables/plot_summaries.txt").toDF("movieId", "summary")
val movieNameIdMappings = spark.read.option("header","false").option("inferSchema","true").option("delimiter","\t").csv("/FileStore/tables/movie_metadata-ab497.tsv").select($"_c0", $"_c2").toDF("movieId", "name")

// Convert summary column data to lowercase
val summary = movieData.withColumn("summary", lower(col("summary")))
// Calculate document count
val docCount = summary.count

// Preprocessing movie data
// Remove URLs
var withUrlRemoved = summary.withColumn("urlRemoved", regexp_replace($"summary", "(f|ht)tp(s?)://\\S+", ""))
// Remove special characters
var noSpecChar = withUrlRemoved.withColumn("nospech", regexp_replace($"urlRemoved", "[^a-z ]", ""));
// Tokenize summary column
val tokenizer = new Tokenizer().setInputCol("nospech").setOutputCol("tokenizedSummary")
val tokenized = tokenizer.transform(noSpecChar)
// Remove stop words
val remover = new StopWordsRemover().setInputCol("tokenizedSummary").setOutputCol("filtered")
var cleanCorpus = remover.transform(tokenized)

// Adding a new column with filtered column array data converted to string
cleanCorpus = cleanCorpus.withColumn("filteredString", concat_ws(" ", $"filtered"))

// Calculating term and document frequency using MapReduce
var testDF = cleanCorpus.select("filtered", "movieId")
val columns = testDF.columns.map(col) :+
      (explode(col("filtered")) as "token")
val unfoldedDocs = testDF.select(columns: _*)
      
var tf_doc = unfoldedDocs.groupBy("movieId", "token")
                         .agg(count("filtered") as "tf")

var df_doc = unfoldedDocs.groupBy("token")
                         .agg(countDistinct("movieId") as "df")

// Function definition for calculating IDF
def calcIDF(docCount:Long, df: Long) : Double = {
  val idf = math.log(((docCount)/(df + 1)).toDouble)
  return idf
}

// UDF for calcIDF function
val calcIdfUdf = udf {df: Long => calcIDF(docCount, df)}

// Calculating IDF
val idfDocs = df_doc.withColumn("idf", calcIdfUdf(col("df")))

// Calculating TF-TDF 
val tfIdfDocs = tf_doc.join(idfDocs, Seq("token"), "left").withColumn("tf_idf", col("tf") * col("idf"))
tfIdfDocs.show(10)

// Accepting user input: single term query string
dbutils.widgets.text("searchInput", "", "Single Search Query:")
var userInput = dbutils.widgets.get("searchInput")

val searchResults = tfIdfDocs.filter($"token" === userInput).orderBy(desc("tf_idf"))

if(searchResults.count == 0) {
  print("No results found")
} else {
  // Top 10 search results for query string
  searchResults.join(movieNameIdMappings, Seq("movieId")).select($"name").show(10, false) 
}

// Accepting multi-word user query input
dbutils.widgets.text("multiSearchInput", "", "Multi Search Query:")
val userInput = dbutils.widgets.get("multiSearchInput")

val queryTokens = userInput.toLowerCase.split(" ")
// MapReduce 
val map = queryTokens.map(s => (s, 1))
val reduce = sc.parallelize(map).reduceByKey((x, y) => x + y).sortBy(_._1)
val queryMap = reduce.collectAsMap

// Cosine Similarity function definition
def cosine_similarity(summary: String) : Double = {
  var total = 0
  var summarySquareSum = 0
  var querySquareSum = 0

  val summaryWords = summary.split(" ").map(word => (word, 1))

  for((word, count) <- summaryWords) {
    var queryWordCount = queryMap.get(word).getOrElse(0).asInstanceOf[Int]
    total = total + queryWordCount
    summarySquareSum += (count * count)
  }
  for ((word, count) <- queryMap) {
    querySquareSum += (count * count)
  }
  var squareRoot = math.sqrt(summarySquareSum * querySquareSum)
  var cosineSimilarity = total / squareRoot
  return cosineSimilarity
}

// UDF for Cosine Similarity function
val udf_cosine_sim = udf {filtered: String => cosine_similarity(filtered)}
// Calculating cosine similarity
val withCosineSim = cleanCorpus.withColumn("cosine_sim", udf_cosine_sim(col("filteredString")))
// Join with movieNameIdMappings to get movie name for every movie Id
val withMovieNames = withCosineSim.join(movieNameIdMappings, Seq("movieId"))
// Fetching top 10 results
val top10Results = withMovieNames.filter(withMovieNames("cosine_sim") > 0).orderBy(desc("cosine_sim")).limit(10)
if(top10Results.count == 0) {
  println("No results found")
} else {
  // Gettings movie names from results
  val queryResults = top10Results.select("name")
  queryResults.show(false)
}
