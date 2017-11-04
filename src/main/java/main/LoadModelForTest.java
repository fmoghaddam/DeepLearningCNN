package main;


import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.FileLabeledSentenceProvider;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class LoadModelForTest {

	public static final String DATA_PATH = "/home/fbm/eclipse-workspace/General Data/RoleTaggerGroundTruth-master/Roles/test/test30/";

	/** Location (local file system) for the Google News vectors. Set this manually. */
	//public static final String WORD_VECTORS_PATH = "/home/fbm/Desktop/CNN_sentence-master/paper_code/GoogleNews-vectors-negative300.bin.gz";
	public static final String WORD_VECTORS_PATH = "/home/fbm/eclipse-workspace/General Data/Google Word2Vec/GoogleNews-vectors-negative300.bin.gz";	


	public static void main(String[] args) {
		// TODO Auto-generated method stub
		File locationToSave = new File("2class.zip"); 
		System.out.println("model loaded");
		try {

			int batchSize = 32;
			int truncateReviewsToLength = 243;
			Random rng = new Random(12345);
			System.out.println("loading word2vec");
			//WordVectors wordVectors = null;
			WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(WORD_VECTORS_PATH));
			DataSetIterator testIter = getDataSetIterator(false, wordVectors, batchSize, truncateReviewsToLength, rng);

			ComputationGraph restored = ModelSerializer.restoreComputationGraph(locationToSave);
			int pos=0;
			int neg = 0;

			String directory = DATA_PATH+"/negative";
			File[] listOfFiles = new File(directory).listFiles();

			double tp = 0;
			double tn = 0;
			double fp = 0;
			double fn = 0;
			
			for (int i = 0; i < listOfFiles.length; i++) {
				final String fileName = listOfFiles[i].getName();
				String entityName;
				final BufferedReader br = new BufferedReader(
						new FileReader(directory+File.separator+fileName));
				while ((entityName = br.readLine()) != null) {
					INDArray featuresFirstNegative = ((CnnSentenceDataSetIterator)testIter).loadSingleSentence(entityName);

					INDArray predictionsFirstNegative = restored.outputSingle(featuresFirstNegative);
					List<String> labels = testIter.getLabels();
					if(predictionsFirstNegative.getDouble(1)>predictionsFirstNegative.getDouble(0)) {
						fp++;
					}else {
						tn++;
					}
				}
			}
			
			System.out.println("precision negative="+(tn/(tn+fp)));
			
			
			for (int i = 0; i < listOfFiles.length; i++) {
				final String fileName = listOfFiles[i].getName();
				String entityName;
				final BufferedReader br = new BufferedReader(
						new FileReader(directory+File.separator+fileName));
				while ((entityName = br.readLine()) != null) {
					INDArray featuresFirstNegative = ((CnnSentenceDataSetIterator)testIter).loadSingleSentence(entityName);

					INDArray predictionsFirstNegative = restored.outputSingle(featuresFirstNegative);
					List<String> labels = testIter.getLabels();
					if(predictionsFirstNegative.getDouble(1)>predictionsFirstNegative.getDouble(0)) {
						tp++;
					}else {
						fn++;
					}
				}
			}
			System.out.println("precision positive="+(tp/(tp+fn)));
			System.err.println("Accuracy= "+(tp+tn)/(tp+tn+fp+fn));
			
//			for (int i = 0; i < listOfFiles.length; i++) {
//				final String fileName = listOfFiles[i].getName();
//				String entityName;
//				final BufferedReader br = new BufferedReader(
//						new FileReader(directory+File.separator+fileName));
//				while ((entityName = br.readLine()) != null) {
//					INDArray featuresFirstNegative = ((CnnSentenceDataSetIterator)testIter).loadSingleSentence(entityName);
//
//					INDArray predictionsFirstNegative = restored.outputSingle(featuresFirstNegative);
//					List<String> labels = testIter.getLabels();
//					if(predictionsFirstNegative.getDouble(1)>predictionsFirstNegative.getDouble(0)) {
//						pos++;
//					}else {
//						neg++;
//					}
//				}
//			}
//
//			System.out.println("Positive = " +pos + " negative = " + neg);
//			System.out.println("Positive = "+(100*pos)/(pos+neg)+"%");
//			System.out.println("Negative ="+(100*neg)/(pos+neg)+"%");
//			
//			directory = DATA_PATH+"/positive";
//			listOfFiles = new File(directory).listFiles();
////			directory = "positive";
////			listOfFiles = new File(directory).listFiles();
////
//			for (int i = 0; i < listOfFiles.length; i++) {
//				final String fileName = listOfFiles[i].getName();
//				String entityName;
//				final BufferedReader br = new BufferedReader(
//						new FileReader(directory+File.separator+fileName));
//				while ((entityName = br.readLine()) != null) {
//					INDArray featuresFirstNegative = ((CnnSentenceDataSetIterator)testIter).loadSingleSentence(entityName);
//
//					INDArray predictionsFirstNegative = restored.outputSingle(featuresFirstNegative);
//					List<String> labels = testIter.getLabels();
//					if(predictionsFirstNegative.getDouble(1)>predictionsFirstNegative.getDouble(0)) {
//						pos++;
//					}else {
//						neg++;
//					}
//				}
//			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	private static DataSetIterator getDataSetIterator(boolean isTraining, WordVectors wordVectors, int minibatchSize,
			int maxSentenceLength, Random rng ){
		String path = FilenameUtils.concat(DATA_PATH, (isTraining ? "" : ""));
		String positiveBaseDir = FilenameUtils.concat(path, "positive");
		String negativeBaseDir = FilenameUtils.concat(path, "negative");

		File filePositive = new File(positiveBaseDir);
		File fileNegative = new File(negativeBaseDir);

		Map<String,List<File>> reviewFilesMap = new HashMap<>();
		reviewFilesMap.put("Positive", Arrays.asList(filePositive.listFiles()));
		reviewFilesMap.put("Negative", Arrays.asList(fileNegative.listFiles()));

		LabeledSentenceProvider sentenceProvider = new FileLabeledSentenceProvider(reviewFilesMap, rng);

		return new CnnSentenceDataSetIterator.Builder()
				.sentenceProvider(sentenceProvider)
				.wordVectors(wordVectors)
				.minibatchSize(minibatchSize)
				.maxSentenceLength(maxSentenceLength)
				.useNormalizedWordVectors(false)
				.build();
	}

}
