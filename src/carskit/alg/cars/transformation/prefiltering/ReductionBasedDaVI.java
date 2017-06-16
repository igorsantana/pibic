package carskit.alg.cars.transformation.prefiltering;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.Table;

import carskit.alg.baseline.cf.ItemKNN;
import carskit.alg.baseline.cf.ItemKNNUnary;
import carskit.alg.cars.transformation.virtualitems.DaVIALL;
import carskit.data.processor.DataSplitter;
import carskit.data.processor.SegmentFinder;
import carskit.data.structure.Segment;
import carskit.data.structure.SparseMatrix;
import carskit.generic.Recommender;
import carskit.main.CARSKit;
import happy.coding.io.LineConfiger;
import happy.coding.io.Logs;
import librec.data.MatrixEntry;
import librec.data.SparseVector;

public class ReductionBasedDaVI extends ReductionBased {

	
	
	public ReductionBasedDaVI(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) throws Exception{
		super(trainMatrix, testMatrix, fold);
		this.algoName = "ReductionBased-DaVI";
	
	}

	
	@Override
	protected Recommender getOuterRecommender(SparseMatrix train, SparseMatrix test, int fold, int condition) throws Exception{
		Logs.warn("DaVI-ALL is going to be used");
		
		int toIgnore = rateDao.getDimensionByConditionId(condition);
		Collection<Integer> cond = new ArrayList<>();
		
		for(int key : rateDao.getDimConditionsList().keySet()) if(key != toIgnore){
			cond.addAll(rateDao.getConditionByDimensionId(key));
		}
		DaVIALL.rateDao = rateDao;
		Recommender model = (new DaVIALL(trainMatrix, testMatrix, "itemknnunary", cond)).getRecommender();
//		model.setItemFrequency(true);
		return model;
	}
}
