package carskit.alg.cars.transformation.virtualitems;

import java.util.ArrayList;
import java.util.Collection;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;

import javax.swing.text.html.HTMLDocument.HTMLReader.IsindexAction;

import java.util.Set;

import org.apache.commons.math3.stat.inference.TTest;


import com.google.common.collect.HashBasedTable;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multimap;
import com.google.common.collect.Table;
import com.google.common.primitives.Doubles;

import carskit.alg.baseline.cf.ItemKNN;
import carskit.alg.baseline.cf.ItemKNNUnary;
import carskit.alg.baseline.cf.SVDPlusPlus;
import carskit.alg.baseline.cf.UserKNN;
import carskit.alg.baseline.ranking.BPR;
import carskit.data.processor.DataDAO;
import carskit.data.processor.DataSplitter;
import carskit.data.structure.SparseMatrix;
import carskit.generic.Recommender;
import carskit.generic.Recommender.Measure;
import librec.data.MatrixEntry;
import librec.data.SparseVector;
import librec.util.Logs;
import librec.util.Stats;


public class DaVIALL {

	public static DataDAO 	rateDao;
	private SparseMatrix 	train, test;
	private String 			recsysName;
	private int 			outerFold;
	private Recommender 	recsys;	
	private Collection<Integer> conditions;
	
	private Recommender getRecommender(SparseMatrix train, SparseMatrix test, int fold){
		Recommender algo = null;
		switch (recsysName) {
			case "itemknn":
				algo = new ItemKNN(train,test, fold);
				break;
			case "itemknnunary":
				algo = new ItemKNNUnary(train, test,fold);
				break;
			case "userknn":
				algo = new UserKNN(train, test, fold);
				break;
			case "bpr":
				algo = new BPR(train, test, fold);
				break;
			case "svd++":
				algo = new SVDPlusPlus(train,test, fold);
				break;

		}
		algo.setResultsOut(false);
		return algo;
	}
	

	private Recommender createDaVIAlgorithm(SparseMatrix train, SparseMatrix test, DataDAO rateDao, int fold){
		DaVI dTrain = new DaVI(train, this.conditions,rateDao);
		DaVI dTest	= new DaVI(test, this.conditions,rateDao);
		
		Recommender algo = getRecommender(dTrain.getMatrix(), dTest.getMatrix(), fold);
		
		algo.setDaVIMappers("davibest",dTrain.getUserUIMapper(), dTrain.getItemUIMapper(), dTest.getUserUIMapper(), dTest.getItemUIMapper());
		algo.setVirtualItems(dTrain.getVirtualItemList());
		return algo;
	}
	
	private Recommender runDaVIAlgo() throws Exception{
		Recommender algo = this.createDaVIAlgorithm(this.train, this.test, rateDao, this.outerFold);
		return algo;
	}
	
	private Recommender runPureAlgo() throws Exception{
		Recommender algo = this.getRecommender(this.train, this.test, this.outerFold);
		return algo;
	}
	
	private void setup_and_run() throws Exception {
		this.recsys = this.conditions.isEmpty() ? runPureAlgo() : runDaVIAlgo(); 
	}
	
	
	public Recommender getRecommender() {
		return this.recsys;
	}
	
	private void populateConditions(){
		for(Integer key : rateDao.getDimConditionsList().keySet()){
			this.conditions.addAll(rateDao.getConditionByDimensionId(key));
		}
	}
	

	public DaVIALL(SparseMatrix train, SparseMatrix test, String recsysName, Collection<Integer> cond) throws Exception {
		this.recsysName = "DaVI-ALL";
		this.train = train;
		this.test = test;
		this.recsysName = recsysName;
		this.conditions = cond;
		this.outerFold = 0;
		if(cond.isEmpty())
			populateConditions();
			
		setup_and_run();
	}

	
}
