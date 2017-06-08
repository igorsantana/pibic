package carskit.data.processor;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.Table;

import carskit.data.structure.Segment;
import carskit.data.structure.SparseMatrix;
import librec.data.MatrixEntry;
import librec.data.SparseVector;
import librec.util.Logs;

public class SegmentFinder {
	/**
	 * rows: User-Item ID (uid,iid) columns: Contexts ID
	 */
	private SparseMatrix train;
	private SparseMatrix test;
	private DataDAO dao;
	private final Map<Integer, Integer> contextCounter = new HashMap<>();
	private final Map<Integer, Integer>      uid_tests 		= new HashMap<>();  

	/**
	 * Constructor for the SegmentFinder class
	 * @param train TrainMatrix Data
	 * @param test  TestMatrix Data
	 * @param rateDao DAO where the information about the matrices.
	 * @param shouldPopulate This parameter is used to control if this class should find the large segments,
	 * or just be used get the segments data.
	 */
	public SegmentFinder(SparseMatrix train, SparseMatrix test, DataDAO rateDao, boolean shouldPopulate) {
		this.train = train;
		this.test = test;
		this.dao = rateDao;
		if(shouldPopulate){
			this.populateCounter();
			this.uidsTests();	
		}
		
	}

	private void uidsTests() {
		for(int uiid : test.rows()){
			int uid = dao.getUserIdFromUI(uiid);
			if(!uid_tests.containsKey(uid)){
				uid_tests.put(uid, 0);
			}
			uid_tests.replace(uid,(uid_tests.get(uid)) + 1);
		}
	}

	private void populateCounter() {
		
		for(MatrixEntry me : this.train){
			
			String[] contexts = dao.getContextId(me.column()).split(",");
			for(int i = 0; i < contexts.length; i++){
				Integer ctx = Integer.parseInt(contexts[i]);
				if(!contextCounter.containsKey(ctx)){
					contextCounter.put(ctx, 0);
				}
				contextCounter.replace(ctx, contextCounter.get(ctx) + 1);
			}
		}
	}
	
	/**
	 * This function is used to do the operations to create a SparseMatrix from a list of uiids that were previously
	 * stored.
	 * @param segment UIIDs that were stored on contextCounter variable.
	 * @return a SparseMatrix of the segment.
	 */
	private SparseMatrix removeTestEntryFromMatrix(SparseMatrix matrix){
		
		SparseMatrix sm = new SparseMatrix(matrix);
			
		for(MatrixEntry me : sm ) if(uid_tests.containsKey(dao.getUserIdFromUI(me.row()))){
			
			Integer ui = me.row();
			
			SparseVector items = sm.row(ui);
			for (int ctx : items.getIndex()) {
				sm.set(ui, ctx, 0.0);
     		}
			
		}
		
		SparseMatrix.reshape(sm);
		
		return sm; 
	}
	/**
	 * Function that is used to find the large segments. A Large Segment is a segment where the number of entries
	 * is higher then the threshold.
	 * @param threshold
	 * @return A List of Large Segments
	 * @throws FileNotFoundException 
	 */
	public List<Segment> getLargeSegments(int threshold) throws FileNotFoundException {

		List<Segment> segments = new ArrayList<>();
		
		for(Integer key : contextCounter.keySet()) {
//			Logs.info("Context {} - Counter {}", dao.getContextConditionId(key), contextCounter.get(key));
			if(contextCounter.get(key) >= threshold){
				SparseMatrix sm = removeTestEntryFromMatrix(getSegmentData(key));
				
				if(contextCounter.get(key) > threshold){
					segments.add(new Segment(sm, dao.getContextConditionId(key), key));
//					writeMatrix(sm,  dao.getContextConditionId(key));
				}
			}	
		}
		
		return segments;
	}
	
	private void writeMatrix(SparseMatrix matrix, String condition){
		try{ 
			PrintWriter pw = new PrintWriter(condition +".txt");
			for(MatrixEntry me : matrix){
				StringBuilder sb  = new StringBuilder();
				String[] contexts = dao.getContextId(me.column()).split(",");
				
				
				
				sb.append("User: " + dao.getUserId(dao.getUserIdFromUI(me.row())));
				sb.append("\t");
				sb.append("Item: " + dao.getItemId(dao.getItemIdFromUI(me.row())));
				sb.append("\t");
				sb.append("Contexts: ");
				for(int i = 0; i < contexts.length; i++){
					sb.append(dao.getContextConditionId(Integer.parseInt(contexts[i])));
					sb.append("\t");
				}
				pw.write(sb.toString() + "\n");
				
			}
			pw.flush();
			pw.close();
		}catch(Exception e){
			
		}
	}
	/**
	 * This function is used to get all the rows that has this Context Condition 
	 * @param condition
	 * @return The SparseMatrix of all the data that matches the condition.
	 * @throws FileNotFoundException 
	 */
	public SparseMatrix getSegmentData(Integer condition) throws FileNotFoundException{
		
		SparseMatrix sm = new SparseMatrix(this.train);
		
		for(MatrixEntry me: this.train ){
			
			Integer ui 	= me.row();
			Integer context = me.column();
			String[] contexts = dao.getContextId(context).split(",");
			
			boolean exclude = true;
			for(String ctx1 : contexts) if(condition.equals(Integer.parseInt(ctx1))){
				exclude = false;
			}
			if(exclude == true){
				SparseVector items = this.train.row(ui);
				for (int ctx : items.getIndex()) {
					sm.set(ui, ctx, 0.0);
         		}
			}
			
		}
		SparseMatrix.reshape(sm);
		
		return sm;

	}

	public DataDAO getDAO() {
		return this.dao;
	}

	public SparseMatrix getTest() {
		return test;
	}

	public SparseMatrix getTrain() {
		return train;
	}
	
	public Collection<Integer> getRemovedUids(){
		return uid_tests.values();
	}
}
