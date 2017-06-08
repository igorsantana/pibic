	package carskit.alg.cars.transformation.prefiltering;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

import com.google.common.collect.Table;

import carskit.alg.baseline.cf.ItemKNN;
import carskit.alg.baseline.cf.ItemKNNUnary;
import carskit.data.processor.DataSplitter;
import carskit.data.processor.SegmentFinder;
import carskit.data.structure.Segment;
import carskit.data.structure.SparseMatrix;
import carskit.generic.Recommender;
import happy.coding.io.LineConfiger;
import happy.coding.io.Logs;
import librec.data.MatrixEntry;
import librec.data.SparseVector;

public class ReductionBased extends Recommender {

	protected int thresholdPercentage;
	
	protected final List<Segment> segments = new ArrayList<>();
	protected final List<Double> metrics   = new ArrayList<>();
	protected int numFolds;
	protected Recommender dsRec;
	protected String rec;
		
	public ReductionBased(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) throws Exception{
		super(trainMatrix, testMatrix, fold);
		
		this.algoName 				= "ReductionBased";

		ReductionBased.algoOptions	= new LineConfiger(cf.getString("recommender"));
		this.thresholdPercentage 	= ReductionBased.algoOptions.getInt("-tp");
		this.numFolds 				= ReductionBased.algoOptions.getInt("-innerfolds");
		this.rec 					= ReductionBased.algoOptions.getString("-traditional");
		
		
		Runnable tDataset = () -> {
			Logs.debug("Começada a execução do modelo inteiro do fold {}", this.fold);
			this.dsRec 	= getRecommender(this.trainMatrix, this.testMatrix, this.fold);
			try {
				this.dsRec.execute();
			} catch (Exception e) {
				e.printStackTrace();
			}
		};

		Runnable tInnerCrossfold = () -> {
			try {
				innerCrossfold();
			} catch (Exception e) {
				e.printStackTrace();
			}
		};
		
		Runnable tOuterRecommenders = () -> {
			for(Segment seg : this.segments){
				try {
					seg.setRecommender(getOuterRecommender(filterMatrix(this.trainMatrix, seg.getData()), this.testMatrix, this.fold, 0));
					seg.getRecommender().execute();
				} catch (Exception e) {
					e.printStackTrace();
				}
				seg.updateMetrics(seg.getRecommender().measures.get(Measure.F110));
			}
		};
		
		SegmentFinder segFinder		= new SegmentFinder(this.trainMatrix, this.testMatrix, rateDao, true);
		this.segments.addAll(segFinder.getLargeSegments(getThreshold()));
		for (Segment seg : this.segments){
			Logs.debug("Segmento encontrado: {}", seg.getName());
		}
		Thread datasetRecommender = new Thread(tDataset);
		Thread innerCrossfold     = new Thread(tInnerCrossfold);
		Thread outerRecommenders  = new Thread(tOuterRecommenders);
		
		datasetRecommender.start();
		innerCrossfold.start();
		innerCrossfold.join();
		Logs.info("Número de segmentos após o innerCrossfold: {}", this.segments.size());
		
		outerRecommenders.start();
		outerRecommenders.join();
		
		this.segments.sort((s1, s2) -> (s1.getMetric() < s2.getMetric()) ? -1 : 1 );
		
		datasetRecommender.join();
	}
	
	Thread threadCreator(Runnable function){
		return new Thread(function);
	}
	
	private int getThreshold(){
		return (int) (((float) this.trainMatrix.size() * this.thresholdPercentage) / 100f);
	}

	private Recommender getRecommender(SparseMatrix train, SparseMatrix test, int fold){
		
		Recommender recsys = null;
		if(this.rec.equals("itemknn")){
			recsys = new ItemKNN(train, test, fold);
		}
		if(this.rec.equals("itemknnunary")){
			recsys = new ItemKNNUnary(train, test, fold);
		}
		recsys.setItemFrequency(true);
		return recsys;
	}
	
	@Override
	public double predict(int u, int j, int c) throws Exception {	
		
		if(this.segments.size() == 0){
			return this.dsRec.recommend(u, j, c);
		}
		
		List<String> contexts = Arrays.asList(rateDao.getContextId(c).split(","))
									.stream().map(ctx -> rateDao.getContextConditionId(Integer.parseInt(ctx)))
									.collect(Collectors.toList());
		
		for(Segment s : this.segments){
			if(contexts.contains(s.getName())){
				return s.getRecommender().recommend(u, j, c);
			}
		}
		return this.dsRec.recommend(u, j, c);
	}

	private SparseMatrix filterMatrix(SparseMatrix toFilter, SparseMatrix toRemove) {
		SparseMatrix newMatrix 		  = new SparseMatrix(toFilter);
		Set<Integer> usersFromSegment = new LinkedHashSet<>();
		
		for(MatrixEntry me : toRemove){
			usersFromSegment.add(rateDao.getUserIdFromUI(me.row()));
		}

		for(MatrixEntry me : newMatrix){
			int useritem = me.row();
			int uid = rateDao.getUserIdFromUI(useritem);
			if(usersFromSegment.contains(uid)){
				
				SparseVector items = newMatrix.row(useritem);
				for (int ctx : items.getIndex()) {
					newMatrix.set(useritem, ctx, 0.0);
         		}
			}
		}
		
		SparseMatrix.reshape(newMatrix);
		
		return newMatrix;
	}
	
	private void innerCrossfold() throws Exception{
		
		DataSplitter ds = new DataSplitter(trainMatrix, this.numFolds);
		for(int i = 1; i <= numFolds; i++){

			SparseMatrix[] TTData 	= ds.getKthFold(i);
			int recFold 			= Integer.parseInt(Integer.toString(this.fold).concat(Integer.toString(i)));
				
			Recommender dataSetRec 	= getRecommender(TTData[0], TTData[1], recFold);
			dataSetRec.execute();
			metrics.add(dataSetRec.measures.get(Measure.F110));
			
			for(Segment seg : this.segments){
				SparseMatrix newTrain = filterMatrix(TTData[0], seg.getData());
				SparseMatrix newTest  = filterMatrix(TTData[1], seg.getData());				
				
				seg.setRecommender(getRecommender(newTrain, newTest, recFold));
				seg.getRecommender().execute();
				seg.addMetric(seg.getRecommender().measures.get(Measure.F110));

			}	

		}
		
		double averageMetric = (metrics.stream().reduce(0.0, Double::sum)) / metrics.size();
		List<Segment> toRemove 			= this.segments.stream().filter(seg -> seg.getMetric() < averageMetric).collect(Collectors.toList());
		this.segments.removeAll(toRemove);
		List<Segment> containedSegments = getContainedSegments(this.segments);
		this.segments.removeAll(containedSegments);
	}	
	
	protected Recommender getOuterRecommender(SparseMatrix train, SparseMatrix test, int fold, int condition) throws Exception{
		Logs.warn("The same Recommender that was used inside is going to be used outside");
		return getRecommender(train, test, fold);
	}
	
	/**
	 * This function is used to check if a matrix is contained within each other, individually.
	 * @param outer Segment's data that every row of it will be checked in the DataTable of the "inner" parameter.
	 * @param inner Segement's data that will have its DataTable checked for the "outer" parameter rows.
	 * @return A boolean value which means that, if a true value is returned, the outer segment is contained within the inner segment. 
	 */
	private boolean isSegmentContained(SparseMatrix outer, SparseMatrix inner){
		Table<Integer, Integer, Double> data = inner.getDataTable();
		
		for(int uiid : outer.rows()){	
			if(!data.containsRow(uiid)) return false;
		}
		return true;
	}
	/**
	 * This function is used to find which segments are contained within each other.
	 * @param segments Segments that'll be verified.
	 * @return A list of names of segments that should be removed.
	 */
	private List<Segment> getContainedSegments(List<Segment> segments){
		List<Segment> toRemove = new ArrayList<>();

		for(int i = 0; i < segments.size(); i++){
			SparseMatrix outer = segments.get(i).getData();
 			for(int j = 0; j < segments.size(); j++) if(i != j){
 				SparseMatrix inner = segments.get(j).getData();
 				if(outer.size() <= inner.size() && isSegmentContained(outer, inner)){
 					toRemove.add(segments.get(i));
 				}
 			}
		}

		return toRemove;
	}
	


}
