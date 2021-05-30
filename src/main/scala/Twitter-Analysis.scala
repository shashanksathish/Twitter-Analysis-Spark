import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{Row, SaveMode, SparkSession}
import org.apache.spark.{SparkConf}

case class Metric(Metric: String, Value: BigDecimal)

object TwitterAnalysis {
  
  def main(args: Array[String]): Unit = {
    if(args.length != 2){
      println("Number of Arguments must be two, Input File, and Output File")
    }

    val spconf = new SparkConf().setAppName("TwitterAnalysis")

    val spark = new SparkSession
      .Builder()
      .config(spconf)
      .getOrCreate()

	  import spark.implicits._

    val inputFile = args(0)
    val outputFile =args(1)
      
    // Reading the input
    val twitter_data = spark.read.option("header","true").option("inferSchema","true").csv(inputFile)
    
    // Selecting desired column
    var twitter_raw_data = twitter_data.select("text","airline_sentiment").toDF("text","sentiment")
  
    // Deleting Null Value
    twitter_raw_data = twitter_raw_data.na.drop(Seq("text","sentiment"))
    
    //Splitting the data
    val twitter_core = twitter_raw_data.randomSplit(Array(0.8,0.2),seed = 14)
    val train_data = twitter_core(0)
    val test_data = twitter_core(1)
    
    //Tokenization, Stopword Removal
    val regexTokenizer = new RegexTokenizer()
                     .setInputCol("text")
                     .setOutputCol("words")
                     .setPattern("\\W+")
                     .setGaps(true)
    val remover = new StopWordsRemover()
             .setInputCol(regexTokenizer.getOutputCol)
             .setOutputCol("filtered")
    val hashingTF = new HashingTF()
             .setInputCol(remover.getOutputCol)
             .setOutputCol("features")
    val indexer = new StringIndexer()
             .setInputCol("sentiment")
             .setOutputCol("label")
    
    //Building Logistic Model
    val lr = new LogisticRegression()
         .setMaxIter(10)
    
    // Building Pipeline
    val pipeline = new Pipeline()
             .setStages(Array(regexTokenizer, remover, indexer, hashingTF, lr))
    
    // Param Building
    val paramGrid = new ParamGridBuilder()
            .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
            .addGrid(lr.regParam, Array(0.1, 0.01, 0.001))
            .build()
    
    val cv = new CrossValidator()
            .setEstimator(pipeline)
            .setEvaluator(new MulticlassClassificationEvaluator)
            .setEstimatorParamMaps(paramGrid)
            .setNumFolds(5)
    
    // Model Training
    val cvModel = cv.fit(train_data)
    
    // Model Testing
    val result = cvModel.bestModel.transform(test_data)
    
    val predictionAndLabels = result.select("prediction","label").map 
    { case Row(label: Double, prediction: Double) => (prediction, label)
    }

    val metrics = new MulticlassMetrics(predictionAndLabels.rdd)
    
    val precision = new Metric("Precision", metrics.weightedPrecision)
    val recall = new Metric("Recall", metrics.weightedRecall)
    val fscore = new Metric("F-Score", metrics.weightedFMeasure)
    val falsepositive = new Metric("False Positive", metrics.weightedFalsePositiveRate)
    val truepositive = new Metric("True Positive", metrics.weightedTruePositiveRate)
    val accuracy = new Metric("Accuracy:", metrics.accuracy)
  
    val eval_DF = Seq(precision, recall, fscore, accuracy).toDF()
    
    eval_DF.coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .format("csv")
      .option("header","true")
      .save(outputFile)
  }
}