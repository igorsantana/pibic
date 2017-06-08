package carskit.alg.cars.transformation.virtualitems;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;

import org.apache.commons.math3.stat.inference.TTest;

import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import com.google.common.primitives.Doubles;

import carskit.alg.baseline.cf.ItemKNN;
import carskit.alg.baseline.cf.ItemKNNUnary;
import carskit.alg.baseline.cf.SVDPlusPlus;
import carskit.alg.baseline.cf.UserKNN;
import carskit.alg.baseline.cf.UserKNNUnary;
import carskit.alg.baseline.ranking.BPR;
import carskit.data.processor.DataDAO;
import carskit.data.processor.DataSplitter;
import carskit.data.structure.SparseMatrix;
import carskit.generic.Recommender;
import carskit.generic.Recommender.Measure;
import happy.coding.io.FileIO;
import librec.util.Logs;
import librec.util.Stats;

public class DaVIBest {
	private SparseMatrix trainMatrix, testMatrix;
	private String recsys_traditional;
	//private int topN;
	private int bestDimension;
	private int numInnerFolds, outerFold;
	private DataDAO rateDao;
	
	private Recommender recsys;
	
	private HashMap<Integer, Double> resDimensions;
	private boolean writeTrainingLog = true;
	private String outerFoldInfo;
	
	public DaVIBest (SparseMatrix trainMatrix, SparseMatrix testMatrix, int outerFold, int numInnerFolds, 
			String recsys_traditional, DataDAO rateDao) throws Exception {

		this.trainMatrix = trainMatrix;
		this.testMatrix = testMatrix;
		this.outerFold = outerFold;
		this.numInnerFolds = numInnerFolds;
		this.recsys_traditional = recsys_traditional;
		this.rateDao = rateDao;

		this.bestDimension = -1;
		this.recsys = null;
		this.resDimensions = new HashMap<Integer,Double>();
		this.outerFoldInfo = this.outerFold > 0 ? " fold [" + outerFold + "]" : "";
		//this.topN = 10; // default TopN to evaluate both two-dimensional and DaVIBest models
		
		runAlgorithm();
	}
	
	/**
	 * 
	 * @return the recommender (traditional or multidimensional) chosen by DaVIBest algorithm
	 */
	public Recommender getRecommender() {
		return this.recsys;
	}
	
	/**
	 * 
	 * DaVIBest algorithm
	 * @throws Exception
	 */
	
	private void runAlgorithm() throws Exception {
		
		Collection<Double> resRec1 = new ArrayList<Double>(); //values of the evaluation by using two-dimensional model (baseline model)
		Collection<Double> resRec2 = new ArrayList<Double>(); //values of the evaluation by using multidimensional model (davibest model)
		
		List<String> davilog = null;
		String toFile = null;
		
		boolean isParallelFold = true; 
		
		// prepare kfold data
		DataSplitter ds = new DataSplitter(trainMatrix, numInnerFolds);
		Thread[] ts = new Thread[numInnerFolds];
		
		// building n two-dimensional models from all folds but one 
		// in user-item data space (traditional rating data)
		Recommender[] algosBase = new Recommender[numInnerFolds];
		for (int i=0 ; i<numInnerFolds ; i++) {
			SparseMatrix[] data = ds.getKthFold(i + 1);
			algosBase[i] = getRecommender(data[0], data[1], i+1);
			
			ts[i] = new Thread(algosBase[i]);
			ts[i].start();
			if (!isParallelFold)
				ts[i].join();
		}
		
		if (isParallelFold)
			for (Thread t : ts)
				t.join();
		
		for (Recommender algo : algosBase) {
			resRec1.add(!algo.measures.get(Measure.F110).isNaN() ? algo.measures.get(Measure.F110) : 0.0);
		}
		
		// record the performance of davibest model on training data (for reference only)
		if (writeTrainingLog) {
			toFile = Recommender.workingPath
                    + String.format("davibest-%s_training%s-top10.log",recsys_traditional, outerFoldInfo); // the output-file name
			FileIO.deleteFile(toFile); // delete possibly old files
		}
		
		FileIO.writeString(toFile, "fold_id,alg,dim,f1_tradrec,f1_davibest,ttest,p\n");
		
		// building n two-dimensional models from all folds but one
		// in user-item-context data space where context is virtual item
		// and store the average performance of each inner fold
		
        ts = new Thread[numInnerFolds];
        
        for (Integer d : rateDao.getDimConditionsList().keySet()) {
        	
        	Recommender[] algosDavi = new Recommender[numInnerFolds];

        	for (int i=0 ; i<numInnerFolds ; i++) {
        		SparseMatrix[] data = ds.getKthFold(i + 1);
        		DaVI daviTrain = new DaVI(data[0], rateDao.getConditionByDimensionId(d), rateDao);
        		DaVI daviTest = new DaVI(data[1], rateDao.getConditionByDimensionId(d), rateDao);
        		
        		algosDavi[i] = getRecommender(daviTrain.getMatrix(), daviTest.getMatrix(), i+1);
        		
        		algosDavi[i].setDaVIMappers("davibest",daviTrain.getUserUIMapper(), daviTrain.getItemUIMapper(),
        				daviTest.getUserUIMapper(), daviTest.getItemUIMapper());
        		
        		algosDavi[i].setVirtualItems(daviTrain.getVirtualItemList());
        		
    			ts[i] = new Thread(algosDavi[i]);
    			ts[i].start();
    			
    			if (!isParallelFold)
    				ts[i].join();

        	}
        	
    		if (isParallelFold)
    			for (Thread t : ts)
    				t.join();
    		
    		
            for (Recommender algo : algosDavi) {
            	resRec2.add(!algo.measures.get(Measure.F110).isNaN() ? algo.measures.get(Measure.F110) : 0.0);
            }
            
            double[] s1 = Doubles.toArray(resRec1);
            double[] s2 = Doubles.toArray(resRec2);
			
            boolean ttest = false;
            
            TTest tt = new TTest();
            double p = tt.tTest(s1, s2);

            //Logs.debug("AVG Model: DaVIBest-"+recsys_traditional+"="+Stats.mean(s2) + " " + recsys_traditional +"="+Stats.mean(s1));
            //Logs.debug("p="+p+"\n");
            
            // check whether there is significant difference when we use the multidimensional model
            // if it is, store the performance and the context dimension from this model
			if (p < 0.05) {		
				ttest=true;
				if (Stats.mean(s2) > Stats.mean(s1)) {
					
					Logs.debug(String.format("DaVIBest-%s with Dimension=\"%s\" improved the baseline model around %.2f%%", recsys_traditional, rateDao.getContextDimensionId(d), 
							(Stats.mean(s2)-Stats.mean(s1))/Stats.mean(s1)));
					resDimensions.put(d, Stats.mean(s2));
				}
			}
			FileIO.writeString(toFile, outerFold + "," + recsys_traditional+"," + rateDao.getContextDimensionId(d)+ "," + Stats.mean(s1) + "," + Stats.mean(s2) + "," + ttest + "," + p + "\n", true);
			//davilog.add(outerFold + "," + recsys_traditional+"," + rateDao.getContextDimensionId(d)+ "," + Stats.mean(s1) + "," + Stats.mean(s2) + "," + ttest + "," + p);
			
			resRec2.clear();
            
        }
        
        resRec1.clear();
        resRec2.clear();
        
        
		//make the selection between the best multidimensional model and the two-dimensional model
		Recommender recsys = null;
		Collection<Integer> bestConditions = getBestDim(resDimensions);
		
		Logs.debug("###### MODEL SELECTION #########");
		if (bestConditions == null) {
			Logs.debug("For fold:{} Recommender={} achieved better performance ", outerFold, recsys_traditional);
			recsys = getRecommender(trainMatrix, testMatrix, outerFold);
			
		}
		else {
			int cond = bestConditions.iterator().next();
			int idDim = rateDao.getDimensionByConditionId(cond);
			
			Logs.debug("For fold:{} Recommender=DaVIBest-{} with Dimension=\"{}\" achieved better performance ", outerFold, recsys_traditional, rateDao.getContextDimensionId(idDim));
			
			DaVI dtr = new DaVI(trainMatrix, bestConditions, rateDao);
			DaVI dts = new DaVI(testMatrix, bestConditions, rateDao);
    		
			recsys = getRecommender(dtr.getMatrix(), dts.getMatrix(), outerFold);
			
			recsys.setDaVIMappers("davibest",dtr.getUserUIMapper(), dtr.getItemUIMapper(), dts.getUserUIMapper(), dts.getItemUIMapper());
    		recsys.setVirtualItems(dtr.getVirtualItemList());
    		
    		this.bestDimension = idDim;
    		recsys.setDaviBestDimension(this.bestDimension);
		}
		
		this.recsys = recsys;

		
	}

	/**
	 * 
	 * @param inforDims dimensions and the correspond contextual conditions
	 * @return an array of contextual condition (e.g location dimension can be have "home" and "cinema" as contextual conditions
	 */
	private Collection<Integer> getBestDim(HashMap<Integer, Double> inforDims)
	{		
		
		Double maxValue=-1.0;
		Integer bestDim=-1;
		
		if (inforDims.size() > 0) {
			for (Integer cond : inforDims.keySet()) {
				if (inforDims.get(cond) > maxValue) {
					maxValue = inforDims.get(cond);
					bestDim = cond;
				}
			}
			return rateDao.getConditionByDimensionId(bestDim);
		}
		else
			return null;

	}
	
	/**
	 * 
	 * @param trainMatrix
	 * @param testMatrix
	 * @param fold
	 * @return the background recommender model used by DaVIBest algorithm
	 * @throws Exception
	 * 
	 * TODO: put more recommenders
	 */
	private Recommender getRecommender(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) throws Exception {
		
		Recommender toReturn = null;
		
		switch(this.recsys_traditional.toLowerCase()) {
			case "userknn":
				toReturn = new UserKNN(trainMatrix, testMatrix, fold); break;
			case "userknnunary":
				toReturn = new UserKNNUnary(trainMatrix, testMatrix, fold); break;
			case "itemknn":
				toReturn = new ItemKNN(trainMatrix, testMatrix, fold); break;
			case "itemknnunary":
				toReturn = new ItemKNNUnary(trainMatrix, testMatrix, fold); break;
			case "svd++":
				toReturn = new SVDPlusPlus(trainMatrix, testMatrix, fold); break;
			case "bpr":
				toReturn = new BPR(trainMatrix, testMatrix, fold); break;
			default:
				throw new Exception("No base recommender is specified to DaVIBest during the generation of candidate models!");
		}
		
		toReturn.setResultsOut(false);
		
		return toReturn;
	}

}
