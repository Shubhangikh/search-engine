// A MapReduce program in Scala to compute word frequency for named entities in a large file

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.annotators.ner.NerConverter
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import org.apache.spark.ml.Pipeline
import scala.collection.mutable.WrappedArray

import org.apache.spark.sql.functions.lower
import org.apache.spark.sql.functions.{col}

// getting the pretrained pipeline for Named Entity Recognition
val pipeline = PretrainedPipeline("entity_recognizer_dl", "en")
// Reading the dracula textbook from Gutenberg website and converting it into dataframe
var testDataset = spark.read
                       .option("header", false)
                       .csv("/FileStore/tables/Dracula.txt")
                       .select($"_c0".as("text"))
// getting all the Named Entity in each line in the textbook
var namedEntities = pipeline.transform(testDataset).select("ner_converter.result")
var removeEmpty = udf((array: Seq[String]) => !array.isEmpty)
var cleanDf = namedEntities.filter(removeEmpty($"result"))
var listOfWords = cleanDf.select("result").rdd.map(_.getSeq[Row](0)).map(y=>y.asInstanceOf[WrappedArray[String]].toSeq.map(x=>x).mkString(","))
val wordCount = listOfWords.map(x=>(x.toLowerCase(),1)).reduceByKey((x,y) => x+y)
val sortedWC = wordCount.sortBy(-_._2)
val dfWithSchema = spark.createDataFrame(sortedWC).toDF("word", "count")

display(dfWithSchema)
