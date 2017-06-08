package carskit.alg.cars.transformation.virtualitems;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.Table;

import carskit.data.processor.DataDAO;
import carskit.data.structure.SparseMatrix;
import happy.coding.io.Logs;
import librec.data.MatrixEntry;

public class DaVI {
	private SparseMatrix trainMatrix; //input multidimensional matrix
	private SparseMatrix trainMatrixExt; //output two-dimensional matrix
	
	private Collection<Integer> conditions; //values for each dimension
	private DataDAO rateDao;
	private HashMap<String,Integer> virtualItemIds; //virtual item ids
	private HashMap<Integer,String> virtualIdItems;
	private HashMap<String,Integer> virtualUIIds; //virtual ui ids
	private HashMap <Integer, Integer> uiUserIds; //ui user ids
	private HashMap <Integer, Integer> uiItemIds; //ui item ids
	
	private HashSet<Integer> userIds;
	private HashSet<Integer> itemIds;
	
	static int startId;
	static int startUIId;
	
	public DaVI(SparseMatrix trainMatrix, Collection<Integer> conditions, DataDAO rateDao) {
		this.trainMatrix = trainMatrix;
		this.conditions = conditions;
		this.rateDao = rateDao;
		
		startId = rateDao.numItems();
		startUIId = rateDao.numUserItems();
		
		userIds = new HashSet<Integer>();
		itemIds = new HashSet<Integer>();
		uiUserIds = new HashMap<Integer,Integer>();
		uiItemIds = new HashMap<Integer,Integer>();
		virtualItemIds = new HashMap<String,Integer>();
		virtualIdItems = new HashMap<Integer,String>();
		virtualUIIds = new HashMap<String,Integer>();
		
		run_davi();
	}
	
	public SparseMatrix getMatrix()
	{
		return this.trainMatrixExt;
	}
	
	public int getUserIdFromUI(int uiid)
	{
		return this.uiUserIds.get(uiid);
	}

	public int getItemIdFromUI(int uiid)
	{
		return this.uiItemIds.get(uiid);
	}
	
	public HashMap<Integer,String> getVirtualItemList()
	{
		return this.virtualIdItems;
	}
	
	public HashMap<Integer, Integer> getUserUIMapper()
	{
		return this.uiUserIds;
	}
	
	public HashMap<Integer, Integer> getItemUIMapper()
	{
		return this.uiItemIds;
	}

	
    protected List<Integer> getConditions(int ctx)
    {
        String context=rateDao.getContextId(ctx);
        String[] cts = context.split(",");
        List<Integer> conds = new ArrayList<>();
        for(String ct:cts)
            conds.add(Integer.valueOf(ct));
        return conds;
    }
    
    public int numItems() {
    	return this.itemIds.size();
    }
    
    public int numUsers() {
    	return this.userIds.size();
    }
	
	private void run_davi()
	{
		Table<Integer, Integer, Double> dataTable = HashBasedTable.create();
		Multimap<Integer, Integer> colMap = HashMultimap.create();
		
		for (MatrixEntry me : trainMatrix) {
			int ui = me.row();
			int ctx = me.column();
			Double rate = me.get();
			
			int u = rateDao.getUserIdFromUI(ui);
			int j = rateDao.getItemIdFromUI(ui);
			
			userIds.add(u);
			itemIds.add(j);
			uiUserIds.put(ui, u);
			uiItemIds.put(ui, j);
			
			dataTable.put(ui, ctx, rate);
			colMap.put(ctx, ui);
			
			for (Integer cond : getConditions(ctx)) {
				if (conditions.contains(cond)) {
					String cc = rateDao.getContextConditionId(cond);
					int newid = virtualItemIds.containsKey(cc) ? virtualItemIds.get(cc) : startId++; //virtual item id
					itemIds.add(newid);
					virtualItemIds.put(cc, newid);
					virtualIdItems.put(newid, cc);
					
					int newui = virtualUIIds.containsKey(rateDao.getUserId(u) + "," + cc) ? virtualUIIds.get(rateDao.getUserId(u) + "," + cc) : startUIId++; //virtual ui id					
					uiUserIds.put(newui, u);
					uiItemIds.put(newui, newid);
					
					virtualUIIds.put(rateDao.getUserId(u) + "," + cc, newui);
					
					dataTable.put(newui, ctx, rate);
					colMap.put(ctx, newui);
				}
			}
			
		}
		
		trainMatrixExt = new SparseMatrix(trainMatrix.numRows()+uiItemIds.size(), trainMatrix.numColumns(), dataTable, colMap);
 	}

}
